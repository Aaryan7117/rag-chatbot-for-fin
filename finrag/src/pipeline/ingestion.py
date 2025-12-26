"""
Ingestion Pipeline Orchestrator
================================

Orchestrates the complete document ingestion workflow:
1. Parse PDF → structured document
2. Extract tables → table store
3. Hybrid chunk → chunks with metadata
4. Generate embeddings → vector store
5. Extract entities → knowledge graph
6. Build BM25 index

All stores use the SAME chunks with SAME chunk_ids.
This is the SINGLE CANONICAL chunking pass.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.document.parser import PDFParser, ParsedDocument
from src.document.chunker import HybridChunker, Chunk
from src.document.table_extractor import TableExtractor, ExtractedTable
from src.stores.vector_store import VectorStore
from src.stores.knowledge_graph import KnowledgeGraph, Entity, EntityType
from src.stores.table_store import TableStore
from src.stores.bm25_index import BM25Index


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    doc_id: str
    filename: str
    success: bool
    
    # Counts
    chunks_created: int = 0
    tables_extracted: int = 0
    entities_extracted: int = 0
    
    # Timing
    processing_time_seconds: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    # Document info
    title: Optional[str] = None
    page_count: int = 0


class EntityExtractor:
    """
    Simple entity extractor for knowledge graph population.
    
    Extracts:
    - Company names
    - Financial metrics
    - Monetary values
    - Dates and years
    
    For production, consider using a proper NER model.
    """
    
    # Patterns for entity extraction
    COMPANY_PATTERN = re.compile(
        r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*'
        r'(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Corporation))\.?)\b'
    )
    
    MONEY_PATTERN = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:(million|billion|M|B))?',
        re.IGNORECASE
    )
    
    PERCENTAGE_PATTERN = re.compile(
        r'([\d.]+)\s*%'
    )
    
    YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
    
    # Financial metric keywords
    METRIC_KEYWORDS = [
        "revenue", "sales", "profit", "income", "earnings",
        "assets", "liabilities", "equity", "margin", "growth",
        "cash flow", "debt", "expenses", "costs"
    ]
    
    def extract_entities(
        self, 
        chunk: Chunk
    ) -> List[Entity]:
        """Extract entities from a chunk."""
        entities = []
        text = chunk.text
        
        # Extract companies
        companies = self.COMPANY_PATTERN.findall(text)
        for company in set(companies):
            entities.append(Entity.create(
                name=company,
                entity_type=EntityType.COMPANY,
                chunk_id=chunk.chunk_id
            ))
        
        # Extract monetary values
        for match in self.MONEY_PATTERN.finditer(text):
            value = match.group(1)
            unit = match.group(2) or ""
            name = f"${value} {unit}".strip()
            entities.append(Entity.create(
                name=name,
                entity_type=EntityType.MONEY,
                chunk_id=chunk.chunk_id
            ))
        
        # Extract percentages
        for match in self.PERCENTAGE_PATTERN.finditer(text):
            entities.append(Entity.create(
                name=f"{match.group(1)}%",
                entity_type=EntityType.PERCENTAGE,
                chunk_id=chunk.chunk_id
            ))
        
        # Extract financial metrics mentioned
        text_lower = text.lower()
        for metric in self.METRIC_KEYWORDS:
            if metric in text_lower:
                entities.append(Entity.create(
                    name=metric,
                    entity_type=EntityType.METRIC,
                    chunk_id=chunk.chunk_id
                ))
        
        return entities


class IngestionPipeline:
    """
    Complete document ingestion pipeline.
    
    Orchestrates:
    1. PDF parsing with structure extraction
    2. Table extraction to SQLite (never vectorized)
    3. Hybrid chunking (single canonical pass)
    4. Vector store population
    5. Knowledge graph population
    6. BM25 index building
    
    All stores reference the SAME chunk_ids.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        table_store: Optional[TableStore] = None,
        bm25_index: Optional[BM25Index] = None,
    ):
        """Initialize with optional pre-configured stores."""
        self._vector_store = vector_store
        self._knowledge_graph = knowledge_graph
        self._table_store = table_store
        self._bm25_index = bm25_index
        
        self._parser = PDFParser()
        self._chunker = HybridChunker(
            max_tokens=settings.chunking.max_chunk_tokens,
            min_tokens=settings.chunking.min_chunk_tokens,
            overlap_tokens=settings.chunking.chunk_overlap_tokens
        )
        self._table_extractor = TableExtractor()
        self._entity_extractor = EntityExtractor()
    
    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store
    
    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        if self._knowledge_graph is None:
            self._knowledge_graph = KnowledgeGraph(
                uri=settings.stores.neo4j_uri,
                user=settings.stores.neo4j_user,
                password=settings.stores.neo4j_password
            )
        return self._knowledge_graph
    
    @property
    def table_store(self) -> TableStore:
        if self._table_store is None:
            self._table_store = TableStore()
        return self._table_store
    
    @property
    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index()
        return self._bm25_index
    
    def ingest(self, pdf_path: str | Path) -> IngestionResult:
        """
        Ingest a PDF document into all stores.
        
        This is the main entry point for ingestion.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            IngestionResult with status and statistics
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        result = IngestionResult(
            doc_id="",
            filename=pdf_path.name,
            success=False
        )
        
        try:
            # Step 1: Parse PDF
            print(f"[Ingestion] Parsing: {pdf_path.name}")
            document = self._parser.parse(pdf_path)
            result.doc_id = document.doc_id
            result.title = document.title
            result.page_count = document.total_pages
            
            # Step 2: Extract tables
            print(f"[Ingestion] Extracting tables...")
            tables = self._table_extractor.extract_tables(document)
            self.table_store.add_tables(tables)
            result.tables_extracted = len(tables)
            
            # Step 3: Chunk document (SINGLE CANONICAL PASS)
            print(f"[Ingestion] Chunking document...")
            chunks = self._chunker.chunk_document(document)
            result.chunks_created = len(chunks)
            print(f"[Ingestion] Created {len(chunks)} chunks")
            
            # Step 4: Link tables to chunks
            self._link_tables_to_chunks(tables, chunks)
            
            # Step 5: Add to Vector Store
            print(f"[Ingestion] Adding to vector store...")
            self.vector_store.add_chunks(chunks)
            
            # Step 6: Add to BM25 Index
            print(f"[Ingestion] Building BM25 index...")
            self.bm25_index.add_chunks(chunks)
            
            # Step 7: Populate Knowledge Graph
            print(f"[Ingestion] Populating knowledge graph...")
            result.entities_extracted = self._populate_knowledge_graph(
                document, chunks, tables
            )
            
            result.success = True
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"[Ingestion] Error: {e}")
            import traceback
            traceback.print_exc()
        
        result.processing_time_seconds = time.time() - start_time
        print(f"[Ingestion] Complete: {result.processing_time_seconds:.2f}s")
        
        return result
    
    def _link_tables_to_chunks(
        self,
        tables: List[ExtractedTable],
        chunks: List[Chunk]
    ) -> None:
        """Link tables to chunks that reference them."""
        table_pattern = re.compile(r'\[TABLE:([^\]]+)\]')
        
        for chunk in chunks:
            table_refs = table_pattern.findall(chunk.text)
            for table_id in table_refs:
                self.table_store.link_table_to_chunk(table_id, chunk.chunk_id)
    
    def _populate_knowledge_graph(
        self,
        document: ParsedDocument,
        chunks: List[Chunk],
        tables: List[ExtractedTable]
    ) -> int:
        """
        Populate knowledge graph with document structure and entities.
        
        Creates:
        - Document node
        - Section nodes
        - Chunk nodes (with summary only, not full text)
        - Entity nodes
        - Table nodes
        - All relationships
        """
        entity_count = 0
        
        # Add document
        self.knowledge_graph.add_document(
            doc_id=document.doc_id,
            title=document.title or document.filename,
            filename=document.filename,
            metadata=document.metadata
        )
        
        # Add sections
        for section in document.sections:
            self.knowledge_graph.add_section(
                section_id=section.section_id,
                doc_id=document.doc_id,
                title=section.title,
                item_number=section.item_number,
                page_start=section.page_start,
                page_end=section.page_end
            )
        
        # Add chunks and extract entities
        for chunk in chunks:
            # Add chunk (summary only)
            summary = chunk.text[:200] if len(chunk.text) > 200 else chunk.text
            self.knowledge_graph.add_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section_id=chunk.section_id,
                summary=summary,
                page_start=chunk.page_start,
                page_end=chunk.page_end
            )
            
            # Extract and add entities
            entities = self._entity_extractor.extract_entities(chunk)
            for entity in entities:
                self.knowledge_graph.add_entity(entity)
                entity_count += 1
            
            # Add table references
            for table_id in chunk.referenced_tables:
                self.knowledge_graph.add_table_reference(
                    table_id=table_id,
                    chunk_id=chunk.chunk_id
                )
        
        return entity_count


def ingest_document(pdf_path: str | Path) -> IngestionResult:
    """
    Convenience function to ingest a document.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        IngestionResult with status and statistics
    """
    pipeline = IngestionPipeline()
    return pipeline.ingest(pdf_path)
