"""
FAISS Vector Store
==================

Dense vector storage using FAISS for semantic similarity search.
Uses bge-large-en-v1.5 embeddings (1024 dimensions).

Design Decisions:
- FAISS over Qdrant for local/offline operation without server
- IVF index for scalability with large document collections
- SQLite sidecar for metadata (FAISS only stores vectors)
- Batch operations for efficiency
"""

import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import sqlite3

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


@dataclass
class VectorSearchResult:
    """Result from vector search with metadata."""
    chunk_id: str
    score: float  # Similarity score (higher = better)
    text: str
    doc_id: str
    section_title: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    metadata: Dict[str, Any] = None


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    
    Architecture:
    - FAISS index stores dense vectors (512d or 1024d)
    - SQLite stores chunk metadata and text
    - Mapping between FAISS internal IDs and chunk_ids
    
    This is ONE of the multi-store components.
    All stores reference the SAME chunks via chunk_id.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            index_path: Directory to store index files
            embedding_model: Model name for embeddings
            device: 'cuda' or 'cpu'
        """
        self.index_path = index_path or settings.paths.faiss_dir
        self.embedding_model_name = embedding_model or settings.models.embedding_model
        self.device = device or settings.models.device
        
        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.faiss_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.db"
        self.mapping_file = self.index_path / "id_mapping.pkl"
        
        # Initialize components
        self._embedding_model = None
        self._index = None
        self._id_to_chunk: Dict[int, str] = {}  # FAISS ID → chunk_id
        self._chunk_to_id: Dict[str, int] = {}  # chunk_id → FAISS ID
        self._next_id = 0
        
        # Load existing index if available
        self._load()
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            print(f"[VectorStore] Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            print(f"[VectorStore] Model loaded, dimension: {self._embedding_model.get_sentence_embedding_dimension()}")
        return self._embedding_model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_model.get_sentence_embedding_dimension()
    
    @property
    def index(self) -> faiss.Index:
        """Get or create FAISS index."""
        if self._index is None:
            # Create new index - using Flat for accuracy
            # For large collections, use IVF: faiss.IndexIVFFlat
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product
            print(f"[VectorStore] Created new FAISS index, dimension: {self.dimension}")
        return self._index
    
    def add_chunks(
        self, 
        chunks: List[Any],  # List of Chunk objects
        batch_size: int = 32
    ) -> int:
        """
        Add chunks to the vector store.
        
        Each chunk is:
        1. Embedded using bge-large-en-v1.5
        2. Added to FAISS index
        3. Metadata stored in SQLite
        
        Args:
            chunks: List of Chunk objects from chunker
            batch_size: Batch size for embedding
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        added_count = 0
        
        # Prepare connection for metadata
        conn = sqlite3.connect(self.metadata_file)
        self._ensure_metadata_table(conn)
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Skip already indexed chunks
                new_chunks = [c for c in batch if c.chunk_id not in self._chunk_to_id]
                
                if not new_chunks:
                    continue
                
                # Generate embeddings
                texts = [c.text for c in new_chunks]
                embeddings = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,  # For cosine similarity via IP
                    show_progress_bar=False
                )
                
                # Add to FAISS
                embeddings = np.array(embeddings).astype('float32')
                start_id = self._next_id
                
                self.index.add(embeddings)
                
                # Update mappings
                for j, chunk in enumerate(new_chunks):
                    faiss_id = start_id + j
                    self._id_to_chunk[faiss_id] = chunk.chunk_id
                    self._chunk_to_id[chunk.chunk_id] = faiss_id
                
                self._next_id += len(new_chunks)
                added_count += len(new_chunks)
                
                # Store metadata
                self._store_metadata(conn, new_chunks)
            
            conn.commit()
            
            # Save index
            self._save()
            
        finally:
            conn.close()
        
        print(f"[VectorStore] Added {added_count} chunks, total: {self.index.ntotal}")
        return added_count
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_doc_id: Optional filter by document
            
        Returns:
            List of VectorSearchResult ordered by similarity
        """
        if self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search FAISS
        # Request more results if filtering
        k = min(top_k * 3 if filter_doc_id else top_k, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k)
        
        # Fetch metadata and build results
        results = []
        conn = sqlite3.connect(self.metadata_file)
        
        try:
            for score, faiss_id in zip(distances[0], indices[0]):
                if faiss_id == -1:  # Invalid result
                    continue
                
                chunk_id = self._id_to_chunk.get(faiss_id)
                if not chunk_id:
                    continue
                
                # Fetch metadata
                metadata = self._get_metadata(conn, chunk_id)
                if not metadata:
                    continue
                
                # Apply filter
                if filter_doc_id and metadata.get("doc_id") != filter_doc_id:
                    continue
                
                results.append(VectorSearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=metadata.get("text", ""),
                    doc_id=metadata.get("doc_id", ""),
                    section_title=metadata.get("section_title"),
                    page_start=metadata.get("page_start", 0),
                    page_end=metadata.get("page_end", 0),
                    metadata=metadata
                ))
                
                if len(results) >= top_k:
                    break
        
        finally:
            conn.close()
        
        return results
    
    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Search by pre-computed embedding.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        embedding = np.array([embedding]).astype('float32')
        k = min(top_k, self.index.ntotal)
        
        distances, indices = self.index.search(embedding, k)
        
        results = []
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue
            chunk_id = self._id_to_chunk.get(faiss_id)
            if chunk_id:
                results.append((chunk_id, float(score)))
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text."""
        return self.embedding_model.encode(
            [text],
            normalize_embeddings=True
        )[0]
    
    def _ensure_metadata_table(self, conn: sqlite3.Connection) -> None:
        """Create metadata table if not exists."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                section_id TEXT,
                section_title TEXT,
                page_start INTEGER,
                page_end INTEGER,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_id ON chunk_metadata(doc_id)
        """)
    
    def _store_metadata(
        self, 
        conn: sqlite3.Connection, 
        chunks: List[Any]
    ) -> None:
        """Store chunk metadata in SQLite."""
        for chunk in chunks:
            conn.execute("""
                INSERT OR REPLACE INTO chunk_metadata 
                (chunk_id, doc_id, text, section_id, section_title, 
                 page_start, page_end, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.text,
                chunk.section_id,
                chunk.section_title,
                chunk.page_start,
                chunk.page_end,
                json.dumps(chunk.metadata) if chunk.metadata else "{}"
            ))
    
    def _get_metadata(
        self, 
        conn: sqlite3.Connection, 
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata."""
        cursor = conn.execute("""
            SELECT chunk_id, doc_id, text, section_id, section_title,
                   page_start, page_end, metadata_json
            FROM chunk_metadata WHERE chunk_id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "text": row[2],
            "section_id": row[3],
            "section_title": row[4],
            "page_start": row[5],
            "page_end": row[6],
            "metadata": json.loads(row[7]) if row[7] else {}
        }
    
    def _save(self) -> None:
        """Save index and mappings to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.faiss_file))
        
        # Save ID mappings
        with open(self.mapping_file, 'wb') as f:
            pickle.dump({
                'id_to_chunk': self._id_to_chunk,
                'chunk_to_id': self._chunk_to_id,
                'next_id': self._next_id
            }, f)
    
    def _load(self) -> None:
        """Load index and mappings from disk."""
        if self.faiss_file.exists():
            self._index = faiss.read_index(str(self.faiss_file))
            print(f"[VectorStore] Loaded FAISS index with {self._index.ntotal} vectors")
        
        if self.mapping_file.exists():
            with open(self.mapping_file, 'rb') as f:
                data = pickle.load(f)
                self._id_to_chunk = data['id_to_chunk']
                self._chunk_to_id = data['chunk_to_id']
                self._next_id = data['next_id']
    
    def clear(self) -> None:
        """Clear the index completely."""
        self._index = None
        self._id_to_chunk = {}
        self._chunk_to_id = {}
        self._next_id = 0
        
        # Remove files
        for f in [self.faiss_file, self.mapping_file, self.metadata_file]:
            if f.exists():
                f.unlink()
        
        print("[VectorStore] Index cleared")
    
    @property
    def count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self._index else 0
