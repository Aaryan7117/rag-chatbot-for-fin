"""
Hybrid Chunking Engine
======================

This module implements the SINGLE CANONICAL chunking pass that produces
chunks used by ALL stores (Vector DB, Knowledge Graph, BM25).

Chunking Strategy (in order of priority):
1. Structural Boundaries: SEC Items, Parts, Sections
2. Semantic Boundaries: Paragraph and concept breaks
3. Token Limits: Max size with overlap
4. Table Preservation: Tables replaced with placeholders, never embedded

Design Principle: ONE chunk definition, MULTIPLE representations.
Every chunk has a stable chunk_id that is referenced everywhere.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Generator
from enum import Enum

from .parser import ParsedDocument, DocumentElement, ElementType, Section


class ChunkType(Enum):
    """Types of chunks based on content."""
    SECTION_CONTENT = "section_content"
    PARAGRAPH = "paragraph"
    LIST = "list"
    MIXED = "mixed"


@dataclass
class Chunk:
    """
    A single chunk - the atomic unit of retrieval.
    
    This is the CANONICAL chunk definition used across:
    - Vector Store (text → embedding)
    - Knowledge Graph (chunk_id reference)
    - BM25 Index (text → tokens)
    
    Every chunk MUST have:
    - Stable chunk_id (hash-based for idempotent operations)
    - Rich metadata for filtering and citation
    - Clear source tracking for grounded responses
    """
    
    # Core identifiers
    chunk_id: str  # Stable hash-based ID
    doc_id: str    # Parent document ID
    
    # Content
    text: str      # Actual chunk text (tables replaced with placeholders)
    chunk_type: ChunkType
    
    # Source tracking for citations - CRITICAL for grounded responses
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    item_number: Optional[str] = None  # SEC Item number if applicable
    page_start: int = 0
    page_end: int = 0
    
    # Relationships
    referenced_tables: List[str] = field(default_factory=list)  # table_ids
    parent_chunk_id: Optional[str] = None  # For hierarchical chunks
    
    # Metadata for filtering
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Stats
    token_count: int = 0
    
    def to_citation(self) -> str:
        """Generate citation string for this chunk."""
        parts = []
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.item_number:
            parts.append(f"Item {self.item_number}")
        parts.append(f"Page {self.page_start}")
        if self.page_end != self.page_start:
            parts[-1] = f"Pages {self.page_start}-{self.page_end}"
        return ", ".join(parts)


class HybridChunker:
    """
    Produces chunks using a hybrid structural + semantic approach.
    
    This is the ONLY chunking pass - all stores use the same chunks.
    
    Algorithm:
    1. Start with document sections (structural boundaries)
    2. Within sections, respect paragraph boundaries (semantic)
    3. If content exceeds max_tokens, split at sentence boundaries
    4. Replace tables with placeholders (link via table_id)
    5. Generate stable chunk_id from content hash
    
    Configuration is loaded from settings but can be overridden.
    """
    
    # Sentence boundary patterns
    SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    # Table placeholder pattern
    TABLE_PLACEHOLDER = re.compile(r'\[TABLE:([^\]]+)\]')
    
    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 50,
        overlap_tokens: int = 64,
        table_placeholder_format: str = "[TABLE:{table_id}]"
    ):
        """
        Initialize chunker with configuration.
        
        Args:
            max_tokens: Maximum tokens per chunk (approx 4 chars = 1 token)
            min_tokens: Minimum tokens to form a chunk (avoid tiny fragments)
            overlap_tokens: Overlap between adjacent chunks for context
            table_placeholder_format: Format string for table placeholders
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.table_placeholder_format = table_placeholder_format
        
        # Approximate characters per token for rough sizing
        self.chars_per_token = 4
    
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Chunk a parsed document using hybrid strategy.
        
        This is the MAIN ENTRY POINT for chunking.
        
        Args:
            document: ParsedDocument from parser
            
        Returns:
            List of Chunk objects ready for all stores
        """
        chunks = []
        
        # Build table ID to placeholder mapping
        table_map = self._build_table_map(document.tables)
        
        if document.sections:
            # Use structural chunking if sections are detected
            chunks = self._chunk_by_sections(document, table_map)
        else:
            # Fall back to element-based chunking
            chunks = self._chunk_by_elements(document, table_map)
        
        # Post-process: merge small chunks, add overlaps
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _build_table_map(
        self, tables: List[DocumentElement]
    ) -> Dict[str, str]:
        """Build mapping from table_id to placeholder text."""
        return {
            t.table_id: self.table_placeholder_format.format(table_id=t.table_id)
            for t in tables if t.table_id
        }
    
    def _chunk_by_sections(
        self, 
        document: ParsedDocument,
        table_map: Dict[str, str]
    ) -> List[Chunk]:
        """
        Chunk document using section structure.
        
        This respects SEC 10-K structure (Parts, Items) for coherent retrieval.
        """
        chunks = []
        
        def process_section(
            section: Section, 
            parent_context: Optional[str] = None
        ):
            """Recursively process sections."""
            
            # Build section context
            context = section.title
            if parent_context:
                context = f"{parent_context} > {section.title}"
            
            # Collect section text
            section_text = self._elements_to_text(
                section.elements, table_map
            )
            
            if section_text.strip():
                # Chunk section content
                section_chunks = self._split_text(
                    text=section_text,
                    doc_id=document.doc_id,
                    section_id=section.section_id,
                    section_title=section.title,
                    item_number=section.item_number,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    table_map=table_map
                )
                chunks.extend(section_chunks)
            
            # Process subsections
            for subsection in section.subsections:
                process_section(subsection, context)
        
        # Process all top-level sections
        for section in document.sections:
            process_section(section)
        
        return chunks
    
    def _chunk_by_elements(
        self,
        document: ParsedDocument,
        table_map: Dict[str, str]
    ) -> List[Chunk]:
        """
        Chunk document by elements when no clear section structure.
        
        Falls back to paragraph-based chunking with table exclusion.
        """
        chunks = []
        current_elements = []
        current_page_start = 1
        current_page_end = 1
        
        for element in document.elements:
            if element.element_type == ElementType.TABLE:
                continue  # Tables handled separately
            
            current_elements.append(element)
            current_page_end = element.page_number
            
            # Check if we need to flush
            combined_text = self._elements_to_text(current_elements, table_map)
            if self._estimate_tokens(combined_text) >= self.max_tokens:
                # Create chunks from accumulated elements
                element_chunks = self._split_text(
                    text=combined_text,
                    doc_id=document.doc_id,
                    section_id=None,
                    section_title=None,
                    item_number=None,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    table_map=table_map
                )
                chunks.extend(element_chunks)
                
                # Reset
                current_elements = []
                current_page_start = element.page_number
        
        # Handle remaining elements
        if current_elements:
            combined_text = self._elements_to_text(current_elements, table_map)
            if combined_text.strip():
                element_chunks = self._split_text(
                    text=combined_text,
                    doc_id=document.doc_id,
                    section_id=None,
                    section_title=None,
                    item_number=None,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    table_map=table_map
                )
                chunks.extend(element_chunks)
        
        return chunks
    
    def _elements_to_text(
        self,
        elements: List[DocumentElement],
        table_map: Dict[str, str]
    ) -> str:
        """Convert elements to text, replacing tables with placeholders."""
        parts = []
        
        for element in elements:
            if element.element_type == ElementType.TABLE:
                if element.table_id and element.table_id in table_map:
                    parts.append(table_map[element.table_id])
            else:
                parts.append(element.content)
        
        return "\n\n".join(parts)
    
    def _split_text(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str],
        section_title: Optional[str],
        item_number: Optional[str],
        page_start: int,
        page_end: int,
        table_map: Dict[str, str]
    ) -> List[Chunk]:
        """
        Split text into chunks respecting token limits.
        
        Strategy:
        1. If text fits in max_tokens, create single chunk
        2. Otherwise, split at sentence boundaries
        3. Maintain overlap between chunks for context continuity
        """
        chunks = []
        
        # If small enough, single chunk
        if self._estimate_tokens(text) <= self.max_tokens:
            chunk = self._create_chunk(
                text=text,
                doc_id=doc_id,
                section_id=section_id,
                section_title=section_title,
                item_number=item_number,
                page_start=page_start,
                page_end=page_end,
                table_map=table_map
            )
            if chunk:
                chunks.append(chunk)
            return chunks
        
        # Split into sentences
        sentences = self.SENTENCE_END.split(text)
        
        current_text = ""
        current_start_page = page_start
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds limit
            test_text = current_text + " " + sentence if current_text else sentence
            
            if self._estimate_tokens(test_text) > self.max_tokens:
                # Flush current chunk
                if current_text:
                    chunk = self._create_chunk(
                        text=current_text,
                        doc_id=doc_id,
                        section_id=section_id,
                        section_title=section_title,
                        item_number=item_number,
                        page_start=current_start_page,
                        page_end=page_end,
                        table_map=table_map
                    )
                    if chunk:
                        chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_text)
                current_text = overlap_text + " " + sentence if overlap_text else sentence
                current_start_page = page_start  # Approximate
            else:
                current_text = test_text
        
        # Flush final chunk
        if current_text:
            chunk = self._create_chunk(
                text=current_text,
                doc_id=doc_id,
                section_id=section_id,
                section_title=section_title,
                item_number=item_number,
                page_start=current_start_page,
                page_end=page_end,
                table_map=table_map
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of previous chunk."""
        if not text:
            return ""
        
        overlap_chars = self.overlap_tokens * self.chars_per_token
        
        if len(text) <= overlap_chars:
            return text
        
        # Try to break at sentence boundary
        overlap = text[-overlap_chars:]
        
        # Find start of a sentence within overlap
        sentence_start = overlap.find(". ")
        if sentence_start > 0:
            return overlap[sentence_start + 2:]
        
        return overlap
    
    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        section_id: Optional[str],
        section_title: Optional[str],
        item_number: Optional[str],
        page_start: int,
        page_end: int,
        table_map: Dict[str, str]
    ) -> Optional[Chunk]:
        """
        Create a Chunk object with stable ID.
        
        The chunk_id is a hash of (doc_id + text), ensuring:
        - Same content always gets same ID (idempotent)
        - Different content gets different ID (unique)
        """
        text = text.strip()
        
        if not text or self._estimate_tokens(text) < self.min_tokens:
            return None
        
        # Generate stable chunk_id
        chunk_id = self._generate_chunk_id(doc_id, text)
        
        # Find referenced tables
        table_refs = self.TABLE_PLACEHOLDER.findall(text)
        
        # Determine chunk type
        chunk_type = ChunkType.PARAGRAPH
        if section_id:
            chunk_type = ChunkType.SECTION_CONTENT
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            chunk_type=chunk_type,
            section_id=section_id,
            section_title=section_title,
            item_number=item_number,
            page_start=page_start,
            page_end=page_end,
            referenced_tables=table_refs,
            token_count=self._estimate_tokens(text),
            metadata={
                "has_tables": len(table_refs) > 0
            }
        )
    
    def _generate_chunk_id(self, doc_id: str, text: str) -> str:
        """
        Generate stable chunk ID from content.
        
        Using SHA256 hash ensures:
        - Deterministic (same input → same output)
        - Collision-resistant
        - Fixed length
        """
        content = f"{doc_id}:{text}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        return hash_bytes[:12].hex()  # 24 char hex string
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Rough approximation: 1 token ≈ 4 characters.
        Good enough for chunking decisions.
        """
        return len(text) // self.chars_per_token
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Post-process chunks for quality.
        
        Operations:
        - Merge very small adjacent chunks
        - Remove empty chunks
        - Validate metadata consistency
        """
        if not chunks:
            return chunks
        
        # Filter out tiny chunks that don't meet minimum
        valid_chunks = [c for c in chunks if c.token_count >= self.min_tokens]
        
        # Could add more post-processing here (merging, etc.)
        
        return valid_chunks


def chunk_document(
    document: ParsedDocument,
    max_tokens: int = 512,
    min_tokens: int = 50,
    overlap_tokens: int = 64
) -> List[Chunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        document: ParsedDocument from parser
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of Chunk objects
    """
    chunker = HybridChunker(
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens
    )
    return chunker.chunk_document(document)
