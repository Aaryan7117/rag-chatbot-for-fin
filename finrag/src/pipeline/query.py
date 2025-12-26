"""
Query Pipeline Orchestrator
============================

Orchestrates the complete query workflow:
1. Process query (normalize, expand, extract entities)
2. Hybrid retrieval (vector + BM25 + KG)
3. Cross-encoder reranking (MANDATORY)
4. Confidence threshold check
5. Context assembly
6. Grounded generation with citations

If no chunks pass threshold → "information not found"
"""

from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.retrieval.query_processor import QueryProcessor, ProcessedQuery
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker, RankedResult
from src.generation.response_builder import ResponseBuilder, RAGResponse


@dataclass
class QueryContext:
    """Context passed through the query pipeline."""
    original_query: str
    processed_query: Optional[ProcessedQuery] = None
    retrieval_results: Optional[List[RetrievalResult]] = None
    ranked_results: Optional[List[RankedResult]] = None
    response: Optional[RAGResponse] = None


class QueryPipeline:
    """
    Complete RAG query pipeline.
    
    Strict Retrieval Flow:
    1. Query normalization and intent detection
    2. Query expansion (synonyms, financial terms)
    3. Hybrid retrieval:
       - Vector DB (semantic)
       - BM25 (keyword)
       - Knowledge Graph (entities)
    4. Merge and deduplicate
    5. Cross-encoder reranking (MANDATORY)
    6. Context assembly:
       - Max 3-5 chunks
       - Prefer section coherence
       - Include tables if referenced
       - Enforce confidence threshold
    7. Grounded generation
    
    If no chunk meets threshold:
    → "This information is not present in the provided documents."
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        response_builder: Optional[ResponseBuilder] = None
    ):
        """Initialize with optional pre-configured components."""
        self._query_processor = QueryProcessor()
        self._retriever = retriever
        self._reranker = reranker
        self._response_builder = response_builder
    
    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever
    
    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker
    
    @property
    def response_builder(self) -> ResponseBuilder:
        if self._response_builder is None:
            self._response_builder = ResponseBuilder()
        return self._response_builder
    
    def query(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
        verbose: bool = False
    ) -> RAGResponse:
        """
        Process a query and generate grounded response.
        
        This is the main entry point for RAG queries.
        
        Args:
            query: User question
            doc_id: Optional filter to specific document
            top_k: Number of chunks to use for generation
            verbose: Print debug information
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        context = QueryContext(original_query=query)
        
        # Step 1: Process query
        if verbose:
            print(f"[Query] Processing: {query[:50]}...")
        
        context.processed_query = self._query_processor.process(query)
        
        if verbose:
            print(f"[Query] Intent: {context.processed_query.intent.value}")
            print(f"[Query] Entities: {context.processed_query.entities}")
            print(f"[Query] Expanded to {len(context.processed_query.expanded_queries)} variants")
        
        # Step 2: Hybrid retrieval
        if verbose:
            print(f"[Query] Running hybrid retrieval...")
        
        # Get more candidates than needed for reranking
        initial_k = settings.retrieval.initial_retrieval_k
        
        context.retrieval_results = self.retriever.retrieve(
            query=query,
            top_k=initial_k,
            filter_doc_id=doc_id,
            include_tables=context.processed_query.wants_table
        )
        
        if verbose:
            print(f"[Query] Retrieved {len(context.retrieval_results)} candidates")
        
        # Step 3: MANDATORY reranking
        if verbose:
            print(f"[Query] Reranking with cross-encoder...")
        
        context.ranked_results = self.reranker.rerank(
            query=query,
            candidates=context.retrieval_results,
            top_k=top_k
        )
        
        if verbose:
            print(f"[Query] {len(context.ranked_results)} chunks passed threshold")
            for i, r in enumerate(context.ranked_results[:3]):
                print(f"  [{i+1}] Score: {r.rerank_score:.3f} - {r.text[:50]}...")
        
        # Step 4: Check confidence threshold
        if not context.ranked_results:
            # No results passed threshold
            if verbose:
                print(f"[Query] No chunks passed confidence threshold")
            
            return RAGResponse(
                answer="This information is not present in the provided documents.",
                citations=[],
                source_chunks=[],
                has_answer=False,
                insufficient_context=True,
                query=query,
                confidence=0.0
            )
        
        # Step 5-6: Build response with grounded generation
        if verbose:
            print(f"[Query] Generating response...")
        
        context.response = self.response_builder.build_response(
            query=query,
            ranked_chunks=context.ranked_results,
            include_tables=context.processed_query.wants_table
        )
        
        if verbose:
            print(f"[Query] Confidence: {context.response.confidence:.3f}")
            print(f"[Query] Has answer: {context.response.has_answer}")
        
        return context.response
    
    def get_context_only(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Get ranked context without generation.
        
        Useful for debugging or when using external LLM.
        """
        # Process and retrieve
        processed = self._query_processor.process(query)
        
        candidates = self.retriever.retrieve(
            query=query,
            top_k=settings.retrieval.initial_retrieval_k,
            filter_doc_id=doc_id
        )
        
        # Rerank
        ranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k
        )
        
        return ranked


def process_query(
    query: str,
    doc_id: Optional[str] = None,
    verbose: bool = False
) -> RAGResponse:
    """
    Convenience function to process a query.
    
    Args:
        query: User question
        doc_id: Optional filter to specific document
        verbose: Print debug information
        
    Returns:
        RAGResponse with answer and citations
    """
    pipeline = QueryPipeline()
    return pipeline.query(query, doc_id, verbose=verbose)
