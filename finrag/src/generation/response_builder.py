"""
Response Builder - Citation Formatting and Validation
=======================================================

Assembles final response with:
- Proper citation formatting
- Table insertion when referenced
- Response validation against context
- Confidence scoring
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.retrieval.reranker import RankedResult
from src.stores.table_store import TableStore
from .llm_engine import LLMEngine, get_llm_engine


@dataclass
class RAGResponse:
    """
    Final RAG response with full metadata.
    
    Includes:
    - Generated answer
    - Citations
    - Source chunks
    - Tables (if referenced)
    - Confidence score
    """
    answer: str
    citations: List[Dict[str, Any]]
    source_chunks: List[RankedResult]
    tables: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Response status
    has_answer: bool = True
    insufficient_context: bool = False
    
    # Debug info
    query: str = ""
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "answer": self.answer,
            "has_answer": self.has_answer,
            "confidence": self.confidence,
            "citations": self.citations,
            "tables": [t.get("table_id") for t in self.tables],
            "source_count": len(self.source_chunks),
            "query": self.query
        }


class ResponseBuilder:
    """
    Builds final RAG responses.
    
    Responsibilities:
    1. Assemble context from ranked chunks
    2. Include tables when referenced
    3. Generate response via LLM
    4. Format citations
    5. Validate and score confidence
    """
    
    # Pattern to detect "not found" responses
    NOT_FOUND_PATTERNS = [
        r"not present in the provided documents",
        r"information is not available",
        r"cannot find",
        r"no information about",
        r"not mentioned in",
        r"does not appear in"
    ]
    
    def __init__(
        self,
        llm_engine: Optional[LLMEngine] = None,
        table_store: Optional[TableStore] = None,
        max_context_chunks: int = 5,
        max_context_tokens: int = 3000
    ):
        """
        Initialize response builder.
        
        Args:
            llm_engine: LLM for generation
            table_store: Store for fetching tables
            max_context_chunks: Maximum chunks in context
            max_context_tokens: Approximate token limit for context
        """
        self._llm_engine = llm_engine
        self._table_store = table_store
        self.max_context_chunks = max_context_chunks
        self.max_context_tokens = max_context_tokens
    
    @property
    def llm_engine(self) -> LLMEngine:
        """Lazy-load LLM engine."""
        if self._llm_engine is None:
            self._llm_engine = get_llm_engine()
        return self._llm_engine
    
    @property
    def table_store(self) -> TableStore:
        """Lazy-load table store."""
        if self._table_store is None:
            self._table_store = TableStore()
        return self._table_store
    
    def build_response(
        self,
        query: str,
        ranked_chunks: List[RankedResult],
        include_tables: bool = True
    ) -> RAGResponse:
        """
        Build complete RAG response.
        
        Args:
            query: User query
            ranked_chunks: Reranked chunks that pass threshold
            include_tables: Whether to include referenced tables
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        import time
        start_time = time.time()
        
        # Handle no results case
        if not ranked_chunks:
            return RAGResponse(
                answer="This information is not present in the provided documents.",
                citations=[],
                source_chunks=[],
                has_answer=False,
                insufficient_context=True,
                query=query,
                confidence=0.0
            )
        
        # Select chunks for context (with section coherence preference)
        selected_chunks = self._select_context_chunks(ranked_chunks)
        
        # Fetch tables if referenced
        tables = []
        if include_tables:
            tables = self._fetch_referenced_tables(selected_chunks)
        
        # Build context string
        context = self._assemble_context(selected_chunks, tables)
        
        # Generate response
        answer = self.llm_engine.generate(
            query=query,
            context=context
        )
        
        # Build citations
        citations = self._build_citations(selected_chunks)
        
        # Check if response indicates no answer
        has_answer = not self._is_not_found_response(answer)
        
        # Calculate confidence
        confidence = self._calculate_confidence(ranked_chunks, has_answer)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            source_chunks=selected_chunks,
            tables=tables,
            confidence=confidence,
            has_answer=has_answer,
            insufficient_context=not has_answer,
            query=query,
            processing_time_ms=processing_time
        )
    
    def _select_context_chunks(
        self,
        chunks: List[RankedResult]
    ) -> List[RankedResult]:
        """
        Select chunks for context with coherence preference.
        
        Strategy:
        - Prefer chunks from same section when possible
        - Limit by count and approximate token budget
        - Maintain diversity of information
        """
        if len(chunks) <= self.max_context_chunks:
            return chunks
        
        selected = [chunks[0]]  # Always include top chunk
        current_section = chunks[0].section_title
        
        # Group by section
        same_section = [c for c in chunks[1:] if c.section_title == current_section]
        other_section = [c for c in chunks[1:] if c.section_title != current_section]
        
        # Add same-section chunks first (for coherence)
        for chunk in same_section:
            if len(selected) >= self.max_context_chunks:
                break
            if self._estimate_tokens(selected + [chunk]) < self.max_context_tokens:
                selected.append(chunk)
        
        # Add other sections for diversity
        for chunk in other_section:
            if len(selected) >= self.max_context_chunks:
                break
            if self._estimate_tokens(selected + [chunk]) < self.max_context_tokens:
                selected.append(chunk)
        
        return selected
    
    def _estimate_tokens(self, chunks: List[RankedResult]) -> int:
        """Estimate token count for chunks."""
        total_chars = sum(len(c.text) for c in chunks)
        return total_chars // 4  # Rough estimate: 4 chars per token
    
    def _fetch_referenced_tables(
        self,
        chunks: List[RankedResult]
    ) -> List[Dict[str, Any]]:
        """Fetch tables referenced in chunks."""
        tables = []
        seen_ids = set()
        
        for chunk in chunks:
            for table_id in chunk.referenced_tables or []:
                if table_id not in seen_ids:
                    table = self.table_store.get_table(table_id)
                    if table:
                        tables.append(table)
                        seen_ids.add(table_id)
        
        return tables
    
    def _assemble_context(
        self,
        chunks: List[RankedResult],
        tables: List[Dict[str, Any]]
    ) -> str:
        """
        Assemble context string for LLM.
        
        Format:
        - Each chunk with citation header
        - Tables formatted in markdown
        """
        parts = []
        
        for i, chunk in enumerate(chunks, 1):
            citation = chunk.to_citation()
            parts.append(f"[Source {i}] ({citation})")
            parts.append(chunk.text)
            parts.append("")  # Empty line
        
        # Add tables if present
        if tables:
            parts.append("--- Referenced Tables ---")
            for table in tables:
                if table.get("caption"):
                    parts.append(f"\n**{table['caption']}** (Page {table.get('page_number', 'N/A')})")
                
                # Format table as markdown
                table_md = self._format_table_md(table)
                if table_md:
                    parts.append(table_md)
                parts.append("")
        
        return "\n".join(parts)
    
    def _format_table_md(self, table: Dict[str, Any]) -> str:
        """Format table as markdown."""
        schema = table.get("schema", [])
        rows = table.get("rows", [])
        
        if not schema or not rows:
            return ""
        
        lines = []
        headers = [col["name"] for col in schema]
        
        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Rows (limit to 20 for context)
        for row in rows[:20]:
            values = [str(row.get(h, ""))[:50] for h in headers]  # Truncate long values
            lines.append("| " + " | ".join(values) + " |")
        
        if len(rows) > 20:
            lines.append(f"*[{len(rows) - 20} additional rows]*")
        
        return "\n".join(lines)
    
    def _build_citations(
        self,
        chunks: List[RankedResult]
    ) -> List[Dict[str, Any]]:
        """Build citation list for response."""
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            citations.append({
                "source_num": i,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_title or "Unknown Section",
                "pages": f"{chunk.page_start}" + (
                    f"-{chunk.page_end}" if chunk.page_end != chunk.page_start else ""
                ),
                "score": round(chunk.rerank_score, 3),
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
        
        return citations
    
    def _is_not_found_response(self, answer: str) -> bool:
        """Check if response indicates information not found."""
        answer_lower = answer.lower()
        return any(
            re.search(pattern, answer_lower, re.IGNORECASE)
            for pattern in self.NOT_FOUND_PATTERNS
        )
    
    def _calculate_confidence(
        self,
        chunks: List[RankedResult],
        has_answer: bool
    ) -> float:
        """
        Calculate response confidence score.
        
        Based on:
        - Rerank scores of source chunks
        - Number of supporting chunks
        - Whether answer was found
        """
        if not has_answer:
            return 0.0
        
        if not chunks:
            return 0.0
        
        # Average rerank score
        avg_score = sum(c.rerank_score for c in chunks) / len(chunks)
        
        # Boost for multiple sources
        source_boost = min(len(chunks) / 3, 1.0) * 0.2
        
        # Boost for diverse sources (different sections)
        sections = set(c.section_title for c in chunks if c.section_title)
        diversity_boost = min(len(sections) / 2, 1.0) * 0.1
        
        confidence = avg_score * 0.7 + source_boost + diversity_boost
        
        return min(confidence, 1.0)


def build_response(
    query: str,
    ranked_chunks: List[RankedResult]
) -> RAGResponse:
    """Convenience function to build response."""
    builder = ResponseBuilder()
    return builder.build_response(query, ranked_chunks)
