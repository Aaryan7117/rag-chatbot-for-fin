"""
Configuration Management for Enterprise RAG System
===================================================

This module provides environment-aware configuration for both local development
and Lightning AI GPU deployment. All settings are centralized here to ensure
consistency across the pipeline.

Design Decision: Using pydantic-settings for type-safe configuration with
environment variable support. This allows seamless switching between local
and cloud deployments without code changes.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class ModelSettings(BaseSettings):
    """
    ML Model Configuration
    
    These models are selected for:
    - bge-large-en-v1.5: Strong performance on financial text, 1024 dimensions
    - bge-reranker-large: Cross-encoder for accurate reranking
    - LLaMA-3-8B-Instruct: Open-source, instruction-following, fits in 16GB with 4-bit
    """
    
    # Embedding Model - produces dense vectors for semantic search
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024
    
    # Reranker Model - MANDATORY cross-encoder for final ranking
    reranker_model: str = "BAAI/bge-reranker-large"
    
    # LLM for Generation - 4-bit quantized for memory efficiency
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_quantization: str = "4bit"  # Options: "4bit", "8bit", "none"
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.0  # Deterministic output for grounded responses
    
    # Device configuration - auto-detect GPU availability
    device: str = Field(default="auto")  # "auto", "cuda", "cpu"
    
    class Config:
        env_prefix = "RAG_MODEL_"


class StoreSettings(BaseSettings):
    """
    Multi-Store Configuration
    
    Architecture requires:
    - Vector Store (FAISS): Dense retrieval
    - Knowledge Graph (Neo4j): Structured relationships - MANDATORY
    - Table Store (SQLite): Structured table data
    - BM25 Index: Sparse retrieval
    """
    
    # FAISS Vector Store
    faiss_index_type: str = "IVF256,Flat"  # IVF for scalability
    faiss_nprobe: int = 16  # Search accuracy vs speed tradeoff
    
    # Neo4j Knowledge Graph - MANDATORY component
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"  # Override in production
    neo4j_database: str = "rag"
    
    # SQLite Table Store
    sqlite_db_name: str = "tables.db"
    
    # BM25 Configuration
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization
    
    class Config:
        env_prefix = "RAG_STORE_"


class ChunkingSettings(BaseSettings):
    """
    Hybrid Chunking Configuration
    
    Strategy: Structure-first, then semantic boundaries
    - Respect SEC 10-K Item boundaries
    - Preserve paragraph integrity
    - Extract tables separately (never embed)
    """
    
    # Chunk size limits
    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 50
    chunk_overlap_tokens: int = 64
    
    # Structural patterns for SEC filings
    sec_item_pattern: str = r"(?i)^ITEM\s+\d+[A-Z]?\."
    section_pattern: str = r"(?i)^(?:PART|SECTION)\s+[IVX\d]+"
    
    # Table handling
    table_placeholder_format: str = "[TABLE:{table_id}]"
    
    class Config:
        env_prefix = "RAG_CHUNK_"


class RetrievalSettings(BaseSettings):
    """
    Retrieval Pipeline Configuration
    
    Implements strict retrieval logic:
    1. Hybrid search (vector + BM25 + KG)
    2. Score fusion with configurable weights
    3. Mandatory reranking
    4. Confidence thresholds for grounded responses
    """
    
    # Retrieval counts
    initial_retrieval_k: int = 20  # Candidates before reranking
    final_top_k: int = 5  # Max chunks after reranking
    
    # Score fusion weights (must sum to 1.0)
    vector_weight: float = 0.5
    bm25_weight: float = 0.3
    kg_weight: float = 0.2
    
    # Confidence thresholds - CRITICAL for zero hallucination
    min_rerank_score: float = 0.3  # Below this = "not found"
    min_final_score: float = 0.5  # Required for inclusion in context
    
    # Context assembly rules
    max_context_chunks: int = 5
    prefer_same_section: bool = True
    include_tables_if_referenced: bool = True
    
    class Config:
        env_prefix = "RAG_RETRIEVAL_"


class PathSettings(BaseSettings):
    """
    File System Paths
    
    Organized structure for:
    - Uploaded documents
    - Processed data
    - Index files
    - Logs
    """
    
    # Base directory - auto-detect from module location
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"
    
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss_index"
    
    @property
    def bm25_dir(self) -> Path:
        return self.data_dir / "bm25_index"
    
    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.uploads_dir,
            self.processed_dir,
            self.faiss_dir,
            self.bm25_dir,
            self.tables_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_prefix = "RAG_PATH_"


class Settings(BaseSettings):
    """
    Master Settings Container
    
    Aggregates all configuration sections for easy access.
    Usage:
        from config.settings import settings
        print(settings.models.embedding_model)
    """
    
    models: ModelSettings = Field(default_factory=ModelSettings)
    stores: StoreSettings = Field(default_factory=StoreSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Debug mode
    debug: bool = False
    
    def initialize(self) -> None:
        """Initialize system - create directories, validate config."""
        self.paths.ensure_directories()
        self._detect_device()
    
    def _detect_device(self) -> None:
        """Auto-detect CUDA availability and set device."""
        if self.models.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.models.device = "cuda"
                    print(f"[Config] GPU detected: {torch.cuda.get_device_name(0)}")
                else:
                    self.models.device = "cpu"
                    print("[Config] No GPU detected, using CPU")
            except ImportError:
                self.models.device = "cpu"
    
    class Config:
        env_prefix = "RAG_"


# Global settings instance
settings = Settings()
