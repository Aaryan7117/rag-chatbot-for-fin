"""
BM25 Sparse Search Index
========================

Lexical/keyword search using BM25 algorithm.
Complements dense vector search for hybrid retrieval.

Why BM25:
- Catches exact keyword matches that embeddings might miss
- Better for rare terms, acronyms, specific numbers
- Fast and memory-efficient
- Important for financial/legal documents with precise terminology
"""

import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import nltk
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class BM25SearchResult:
    """Result from BM25 search."""
    chunk_id: str
    score: float
    text: str
    doc_id: str


class BM25Index:
    """
    BM25 sparse index for keyword search.
    
    Works alongside vector store for hybrid retrieval.
    Uses the SAME chunks with SAME chunk_ids.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 index.
        
        Args:
            index_path: Directory to store index files
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.index_path = index_path or settings.paths.bm25_dir
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.k1 = k1
        self.b = b
        
        # File paths
        self.index_file = self.index_path / "bm25_index.pkl"
        self.corpus_file = self.index_path / "corpus.pkl"
        
        # Index state
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []  # Tokenized documents
        self._chunk_ids: List[str] = []  # chunk_id for each corpus entry
        self._chunk_texts: Dict[str, str] = {}  # chunk_id -> text
        self._chunk_docs: Dict[str, str] = {}  # chunk_id -> doc_id
        
        # Stopwords for tokenization
        self._stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Load existing index
        self._load()
    
    def add_chunks(self, chunks: List[Any]) -> int:
        """
        Add chunks to the BM25 index.
        
        Args:
            chunks: List of Chunk objects from chunker
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        added = 0
        for chunk in chunks:
            if chunk.chunk_id in self._chunk_texts:
                continue  # Skip already indexed
            
            # Tokenize text
            tokens = self._tokenize(chunk.text)
            
            if tokens:
                self._corpus.append(tokens)
                self._chunk_ids.append(chunk.chunk_id)
                self._chunk_texts[chunk.chunk_id] = chunk.text
                self._chunk_docs[chunk.chunk_id] = chunk.doc_id
                added += 1
        
        if added > 0:
            # Rebuild BM25 index
            self._bm25 = BM25Okapi(self._corpus, k1=self.k1, b=self.b)
            self._save()
        
        print(f"[BM25Index] Added {added} chunks, total: {len(self._corpus)}")
        return added
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None
    ) -> List[BM25SearchResult]:
        """
        Search for chunks matching query.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_doc_id: Optional filter by document
            
        Returns:
            List of BM25SearchResult ordered by score
        """
        if not self._bm25 or not self._corpus:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Pair with chunk_ids and sort
        scored_chunks = list(zip(self._chunk_ids, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for chunk_id, score in scored_chunks:
            if score <= 0:
                continue
            
            # Apply filter
            if filter_doc_id and self._chunk_docs.get(chunk_id) != filter_doc_id:
                continue
            
            results.append(BM25SearchResult(
                chunk_id=chunk_id,
                score=float(score),
                text=self._chunk_texts.get(chunk_id, ""),
                doc_id=self._chunk_docs.get(chunk_id, "")
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Preprocessing:
        - Lowercase
        - Word tokenization
        - Remove stopwords
        - Remove short tokens
        """
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Filter
        tokens = [
            t for t in tokens
            if len(t) > 2 and  # Remove short tokens
            t not in self._stopwords and  # Remove stopwords
            t.isalnum()  # Remove punctuation-only
        ]
        
        return tokens
    
    def _save(self) -> None:
        """Save index to disk."""
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'corpus': self._corpus,
                'chunk_ids': self._chunk_ids,
                'chunk_texts': self._chunk_texts,
                'chunk_docs': self._chunk_docs,
                'k1': self.k1,
                'b': self.b
            }, f)
    
    def _load(self) -> None:
        """Load index from disk."""
        if not self.index_file.exists():
            return
        
        try:
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
            
            self._corpus = data['corpus']
            self._chunk_ids = data['chunk_ids']
            self._chunk_texts = data['chunk_texts']
            self._chunk_docs = data['chunk_docs']
            self.k1 = data.get('k1', 1.5)
            self.b = data.get('b', 0.75)
            
            if self._corpus:
                self._bm25 = BM25Okapi(self._corpus, k1=self.k1, b=self.b)
            
            print(f"[BM25Index] Loaded index with {len(self._corpus)} chunks")
        except Exception as e:
            print(f"[BM25Index] Failed to load index: {e}")
    
    def clear(self) -> None:
        """Clear the index."""
        self._bm25 = None
        self._corpus = []
        self._chunk_ids = []
        self._chunk_texts = {}
        self._chunk_docs = {}
        
        if self.index_file.exists():
            self.index_file.unlink()
        
        print("[BM25Index] Index cleared")
    
    @property
    def count(self) -> int:
        """Get number of indexed chunks."""
        return len(self._corpus)
