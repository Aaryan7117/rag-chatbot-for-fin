"""
Cross-Encoder Reranker - MANDATORY Component
=============================================

This is a REQUIRED component. The system is INVALID without reranking.

Reranking takes retrieval candidates and re-scores them using a
cross-encoder model that jointly encodes query and document.

Why this matters:
- Bi-encoders (used for initial retrieval) are fast but approximate
- Cross-encoders are slower but much more accurate
- Reranking catches subtle semantic matches that bi-encoders miss
- Critical for zero-hallucination: only well-matched chunks survive

Model: bge-reranker-large (or equivalent cross-encoder)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

# Cross-encoder import
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from .hybrid_retriever import RetrievalResult


@dataclass 
class RankedResult:
    """Result after cross-encoder reranking."""
    chunk_id: str
    text: str
    doc_id: str
    
    # Reranker score (0-1, higher = better)
    rerank_score: float
    
    # Original scores preserved
    combined_score: float
    vector_score: float
    bm25_score: float
    kg_score: float
    
    # Source tracking
    section_title: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    sources: List[str] = None
    referenced_tables: List[str] = None
    
    # Confidence flag
    passes_threshold: bool = True
    
    def to_citation(self) -> str:
        """Generate citation string."""
        parts = []
        if self.section_title:
            title = self.section_title[:50]
            parts.append(f"Section: {title}")
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                parts.append(f"Pages {self.page_start}-{self.page_end}")
            else:
                parts.append(f"Page {self.page_start}")
        return ", ".join(parts) if parts else "Unknown location"


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for final ranking.
    
    MANDATORY COMPONENT - system is invalid without this.
    
    Process:
    1. Take retrieval candidates
    2. Score each (query, chunk) pair with cross-encoder
    3. Apply confidence threshold
    4. Return top-k that pass threshold
    
    If no candidates pass threshold, retrieval fails gracefully
    with "information not found" response.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        min_score: Optional[float] = None
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: 'cuda' or 'cpu'
            min_score: Minimum score to pass (0-1)
        """
        self.model_name = model_name or settings.models.reranker_model
        self.device = device or settings.models.device
        self.min_score = min_score or settings.retrieval.min_rerank_score
        
        self._model = None
    
    @property
    def model(self):
        """Lazy-load cross-encoder model."""
        if self._model is None:
            if not HAS_CROSS_ENCODER:
                raise ImportError(
                    "sentence-transformers required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
            
            print(f"[Reranker] Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            print(f"[Reranker] Model loaded on {self.device}")
        
        return self._model
    
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank candidates using cross-encoder.
        
        This is the MANDATORY reranking step.
        
        Args:
            query: User query
            candidates: Retrieval results to rerank
            top_k: Number of results to return
            
        Returns:
            List of RankedResult ordered by rerank_score
            Only results passing threshold are included
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, c.text) for c in candidates]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Normalize scores to 0-1 range using sigmoid if needed
        scores = self._normalize_scores(scores)
        
        # Build ranked results
        ranked = []
        for candidate, score in zip(candidates, scores):
            passes = score >= self.min_score
            
            ranked.append(RankedResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                doc_id=candidate.doc_id,
                rerank_score=float(score),
                combined_score=candidate.combined_score,
                vector_score=candidate.vector_score,
                bm25_score=candidate.bm25_score,
                kg_score=candidate.kg_score,
                section_title=candidate.section_title,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                sources=candidate.sources or [],
                referenced_tables=candidate.referenced_tables or [],
                passes_threshold=passes
            ))
        
        # Sort by rerank score
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Filter by threshold and limit
        passing = [r for r in ranked if r.passes_threshold]
        
        return passing[:top_k]
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize cross-encoder scores to 0-1 range.
        
        Different cross-encoders output different score ranges.
        We use sigmoid normalization for consistency.
        """
        # Most cross-encoders output logits that need sigmoid
        # Check if scores are already in reasonable range
        if scores.min() >= 0 and scores.max() <= 1:
            return scores
        
        # Apply sigmoid for normalization
        return 1 / (1 + np.exp(-scores))
    
    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-text pair."""
        score = self.model.predict([(query, text)])[0]
        return float(self._normalize_scores(np.array([score]))[0])


def rerank(
    query: str,
    candidates: List[RetrievalResult],
    top_k: int = 5
) -> List[RankedResult]:
    """Convenience function for reranking."""
    reranker = CrossEncoderReranker()
    return reranker.rerank(query, candidates, top_k)
