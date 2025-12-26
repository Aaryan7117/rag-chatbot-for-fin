"""
Tests for Enterprise RAG System
================================

Basic tests for pipeline components.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestQueryProcessor:
    """Tests for query processing."""
    
    def test_normalization(self):
        from src.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        result = processor.process("  What is the REVENUE?  ")
        
        assert result.normalized == "what is the revenue?"
    
    def test_intent_detection_factual(self):
        from src.retrieval.query_processor import QueryProcessor, QueryIntent
        
        processor = QueryProcessor()
        result = processor.process("What are the risk factors?")
        
        assert result.intent == QueryIntent.FACTUAL
    
    def test_intent_detection_comparison(self):
        from src.retrieval.query_processor import QueryProcessor, QueryIntent
        
        processor = QueryProcessor()
        result = processor.process("Compare revenue vs last year")
        
        assert result.intent == QueryIntent.COMPARISON
    
    def test_intent_detection_table(self):
        from src.retrieval.query_processor import QueryProcessor, QueryIntent
        
        processor = QueryProcessor()
        result = processor.process("Show me the breakdown by segment")
        
        assert result.intent == QueryIntent.TABLE_LOOKUP
    
    def test_query_expansion(self):
        from src.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        result = processor.process("What is the revenue?")
        
        # Should have expanded queries with synonyms
        assert len(result.expanded_queries) >= 1
        assert result.normalized in result.expanded_queries


class TestChunker:
    """Tests for hybrid chunking."""
    
    def test_chunk_id_stability(self):
        from src.document.chunker import HybridChunker
        
        chunker = HybridChunker()
        
        # Same input should produce same chunk_id
        id1 = chunker._generate_chunk_id("doc1", "Hello world")
        id2 = chunker._generate_chunk_id("doc1", "Hello world")
        
        assert id1 == id2
    
    def test_chunk_id_uniqueness(self):
        from src.document.chunker import HybridChunker
        
        chunker = HybridChunker()
        
        id1 = chunker._generate_chunk_id("doc1", "Text A")
        id2 = chunker._generate_chunk_id("doc1", "Text B")
        
        assert id1 != id2
    
    def test_token_estimation(self):
        from src.document.chunker import HybridChunker
        
        chunker = HybridChunker()
        
        text = "This is a test sentence."
        tokens = chunker._estimate_tokens(text)
        
        # ~4 chars per token
        assert tokens == len(text) // 4


class TestTableExtractor:
    """Tests for table extraction."""
    
    def test_column_type_number(self):
        from src.document.table_extractor import TableExtractor, ColumnType
        
        extractor = TableExtractor()
        
        samples = ["123", "456.78", "1,234"]
        col_type = extractor._infer_column_type(samples)
        
        assert col_type == ColumnType.NUMBER
    
    def test_column_type_currency(self):
        from src.document.table_extractor import TableExtractor, ColumnType
        
        extractor = TableExtractor()
        
        samples = ["$1,234", "$5.6 million", "$100"]
        col_type = extractor._infer_column_type(samples)
        
        assert col_type == ColumnType.CURRENCY
    
    def test_column_type_percentage(self):
        from src.document.table_extractor import TableExtractor, ColumnType
        
        extractor = TableExtractor()
        
        samples = ["10%", "25.5%", "100%"]
        col_type = extractor._infer_column_type(samples)
        
        assert col_type == ColumnType.PERCENTAGE


class TestResponseBuilder:
    """Tests for response building."""
    
    def test_not_found_detection(self):
        from src.generation.response_builder import ResponseBuilder
        
        builder = ResponseBuilder()
        
        assert builder._is_not_found_response(
            "This information is not present in the provided documents."
        )
        
        assert not builder._is_not_found_response(
            "The revenue was $1 billion."
        )
    
    def test_confidence_calculation(self):
        from src.generation.response_builder import ResponseBuilder
        from src.retrieval.reranker import RankedResult
        
        builder = ResponseBuilder()
        
        # Mock chunks
        chunks = [
            RankedResult(
                chunk_id="1", text="test", doc_id="doc",
                rerank_score=0.8, combined_score=0.7,
                vector_score=0.7, bm25_score=0.5, kg_score=0.0
            )
        ]
        
        confidence = builder._calculate_confidence(chunks, has_answer=True)
        
        assert 0 <= confidence <= 1
        assert confidence > 0


class TestSettings:
    """Tests for configuration."""
    
    def test_settings_initialization(self):
        from config.settings import Settings
        
        settings = Settings()
        
        assert settings.models.embedding_model == "BAAI/bge-large-en-v1.5"
        assert settings.retrieval.max_context_chunks == 5
        assert settings.retrieval.min_rerank_score == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
