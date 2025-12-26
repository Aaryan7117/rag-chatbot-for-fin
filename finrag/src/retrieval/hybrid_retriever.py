"""
Hybrid Retrieval Orchestrator
==============================

Orchestrates multi-path retrieval across all stores:
1. Vector DB (dense semantic search)
2. BM25 (sparse keyword search)
3. Knowledge Graph (entity-based traversal)

Results are merged using Reciprocal Rank Fusion (RRF)
and passed to reranking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.stores.vector_store import VectorStore, VectorSearchResult
from src.stores.bm25_index import BM25Index, BM25SearchResult
from src.stores.knowledge_graph import KnowledgeGraph
from src.stores.table_store import TableStore
from .query_processor import QueryProcessor, ProcessedQuery


@dataclass
class RetrievalResult:
    """
    Result from hybrid retrieval before reranking.
    
    Contains chunk info and scores from different sources.
    """
    chunk_id: str
    text: str
    doc_id: str
    
    # Source tracking for citations
    section_title: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    
    # Scores from different sources
    vector_score: float = 0.0
    bm25_score: float = 0.0
    kg_score: float = 0.0
    
    # Fused score
    combined_score: float = 0.0
    
    # Retrieval source tracking
    sources: List[str] = field(default_factory=list)
    
    # Referenced tables (if any)
    referenced_tables: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_citation(self) -> str:
        """Generate citation string."""
        parts = []
        if self.section_title:
            # Clean up section title
            title = self.section_title[:50]
            parts.append(f"Section: {title}")
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                parts.append(f"Pages {self.page_start}-{self.page_end}")
            else:
                parts.append(f"Page {self.page_start}")
        return ", ".join(parts) if parts else "Unknown location"


class HybridRetriever:
    """
    Orchestrates retrieval across Vector DB, BM25, and Knowledge Graph.
    
    Retrieval Strategy:
    1. Process query (normalize, expand, extract entities)
    2. Run parallel searches:
       - Vector search with query embedding
       - BM25 search with query tokens
       - KG traversal if entities detected
    3. Merge results using Reciprocal Rank Fusion
    4. Deduplicate by chunk_id
    5. Return candidates for reranking
    
    Configuration is loaded from settings.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        table_store: Optional[TableStore] = None,
    ):
        """
        Initialize retriever with stores.
        
        Stores are created on-demand if not provided.
        """
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._knowledge_graph = knowledge_graph
        self._table_store = table_store
        
        self._query_processor = QueryProcessor()
        
        # Fusion weights from settings
        self.vector_weight = settings.retrieval.vector_weight
        self.bm25_weight = settings.retrieval.bm25_weight
        self.kg_weight = settings.retrieval.kg_weight
    
    @property
    def vector_store(self) -> VectorStore:
        """Lazy-load vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store
    
    @property
    def bm25_index(self) -> BM25Index:
        """Lazy-load BM25 index."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index()
        return self._bm25_index
    
    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """Lazy-load knowledge graph."""
        if self._knowledge_graph is None:
            self._knowledge_graph = KnowledgeGraph(
                uri=settings.stores.neo4j_uri,
                user=settings.stores.neo4j_user,
                password=settings.stores.neo4j_password
            )
        return self._knowledge_graph
    
    @property
    def table_store(self) -> TableStore:
        """Lazy-load table store."""
        if self._table_store is None:
            self._table_store = TableStore()
        return self._table_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        include_tables: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve candidate chunks for a query.
        
        This is the main entry point for retrieval.
        Returns candidates that should be passed to reranker.
        
        Args:
            query: User query text
            top_k: Number of candidates to return
            filter_doc_id: Optional filter by document
            include_tables: Whether to fetch referenced tables
            
        Returns:
            List of RetrievalResult ordered by combined score
        """
        # Step 1: Process query
        processed = self._query_processor.process(query)
        
        # Step 2: Run searches in parallel (conceptually)
        # Vector search
        vector_results = self._vector_search(processed, top_k * 2, filter_doc_id)
        
        # BM25 search
        bm25_results = self._bm25_search(processed, top_k * 2, filter_doc_id)
        
        # KG traversal (if entities found)
        kg_chunk_ids = []
        if processed.entities:
            kg_chunk_ids = self._kg_search(processed.entities)
        
        # Step 3: Merge and fuse scores
        merged = self._merge_results(vector_results, bm25_results, kg_chunk_ids)
        
        # Step 4: Deduplicate (already done during merge)
        
        # Step 5: Get table references if needed
        if include_tables:
            merged = self._add_table_references(merged, processed)
        
        # Sort by combined score
        merged.sort(key=lambda x: x.combined_score, reverse=True)
        
        return merged[:top_k]
    
    def _vector_search(
        self,
        processed: ProcessedQuery,
        top_k: int,
        filter_doc_id: Optional[str]
    ) -> List[VectorSearchResult]:
        """Run vector similarity search."""
        all_results = []
        
        # Search with all query variants
        for query in processed.expanded_queries[:3]:  # Use up to 3 variants
            results = self.vector_store.search(
                query=query,
                top_k=top_k,
                filter_doc_id=filter_doc_id
            )
            all_results.extend(results)
        
        # If no expanded queries, use original
        if not all_results:
            all_results = self.vector_store.search(
                query=processed.normalized,
                top_k=top_k,
                filter_doc_id=filter_doc_id
            )
        
        return all_results
    
    def _bm25_search(
        self,
        processed: ProcessedQuery,
        top_k: int,
        filter_doc_id: Optional[str]
    ) -> List[BM25SearchResult]:
        """Run BM25 keyword search."""
        all_results = []
        
        # Search with expanded queries
        for query in processed.expanded_queries[:3]:
            results = self.bm25_index.search(
                query=query,
                top_k=top_k,
                filter_doc_id=filter_doc_id
            )
            all_results.extend(results)
        
        if not all_results:
            all_results = self.bm25_index.search(
                query=processed.normalized,
                top_k=top_k,
                filter_doc_id=filter_doc_id
            )
        
        return all_results
    
    def _kg_search(self, entities: List[str]) -> List[str]:
        """Traverse knowledge graph for entity-related chunks."""
        return self.knowledge_graph.traverse_from_entities(
            entity_names=entities,
            max_hops=2
        )
    
    def _merge_results(
        self,
        vector_results: List[VectorSearchResult],
        bm25_results: List[BM25SearchResult],
        kg_chunk_ids: List[str]
    ) -> List[RetrievalResult]:
        """
        Merge results from all sources using Reciprocal Rank Fusion (RRF).
        
        RRF Score = sum(1 / (k + rank)) for each source
        Where k is a constant (typically 60).
        """
        k = 60  # RRF constant
        
        # Track chunk data and scores
        chunk_map: Dict[str, RetrievalResult] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk_id
            
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=result.text,
                    doc_id=result.doc_id,
                    section_title=result.section_title,
                    page_start=result.page_start,
                    page_end=result.page_end,
                    metadata=result.metadata or {}
                )
            
            rr = chunk_map[chunk_id]
            rr.vector_score = max(rr.vector_score, result.score)
            rr.combined_score += self.vector_weight * (1 / (k + rank))
            if "vector" not in rr.sources:
                rr.sources.append("vector")
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk_id
            
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=result.text,
                    doc_id=result.doc_id
                )
            
            rr = chunk_map[chunk_id]
            rr.bm25_score = max(rr.bm25_score, result.score)
            rr.combined_score += self.bm25_weight * (1 / (k + rank))
            if "bm25" not in rr.sources:
                rr.sources.append("bm25")
        
        # Process KG results (ranked by appearance order)
        for rank, chunk_id in enumerate(kg_chunk_ids, start=1):
            if chunk_id not in chunk_map:
                # Need to fetch chunk data
                # This is a fallback when chunk only found via KG
                chunk_map[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text="",  # Will need to fetch
                    doc_id=""
                )
            
            rr = chunk_map[chunk_id]
            rr.kg_score = 1.0  # Binary presence
            rr.combined_score += self.kg_weight * (1 / (k + rank))
            if "kg" not in rr.sources:
                rr.sources.append("kg")
        
        return list(chunk_map.values())
    
    def _add_table_references(
        self,
        results: List[RetrievalResult],
        processed: ProcessedQuery
    ) -> List[RetrievalResult]:
        """Add table references if chunks reference tables."""
        import re
        table_pattern = re.compile(r'\[TABLE:([^\]]+)\]')
        
        for result in results:
            # Find table references in text
            table_ids = table_pattern.findall(result.text)
            result.referenced_tables = table_ids
            
            # If query wants tables, boost chunks with tables
            if processed.wants_table and table_ids:
                result.combined_score *= 1.2  # 20% boost
        
        return results
    
    def get_tables_for_result(
        self, 
        result: RetrievalResult
    ) -> List[Dict[str, Any]]:
        """Fetch actual table data for a result."""
        tables = []
        for table_id in result.referenced_tables:
            table = self.table_store.get_table(table_id)
            if table:
                tables.append(table)
        return tables


def retrieve(
    query: str,
    top_k: int = 20,
    filter_doc_id: Optional[str] = None
) -> List[RetrievalResult]:
    """Convenience function for retrieval."""
    retriever = HybridRetriever()
    return retriever.retrieve(query, top_k, filter_doc_id)
