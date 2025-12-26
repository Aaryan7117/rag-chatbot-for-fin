# Enterprise RAG System

A high-accuracy, enterprise-grade Retrieval-Augmented Generation system for SEC 10-K filings, financial reports, and legal PDFs.

## Features

- **Zero Hallucination**: All answers grounded in document context with citations
- **Hybrid Chunking**: Structure-aware + semantic chunking in a single pass
- **Multi-Store Architecture**: Vector DB (FAISS), Knowledge Graph (Neo4j), Table Store (SQLite), BM25 Index
- **Mandatory Reranking**: Cross-encoder reranking for accurate retrieval
- **Citation-Grounded Generation**: LLaMA-3-8B-Instruct with 4-bit quantization

## Quick Start

### 1. Install Dependencies

```bash
cd e:\ragprod
pip install -r requirements.txt
```

### 2. (Optional) Set Up Neo4j

For full knowledge graph functionality, install and start Neo4j:

```bash
# Using Docker
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

The system falls back to in-memory mode if Neo4j is unavailable.

### 3. Start the API Server

```bash
cd e:\ragprod
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

### 4. Upload and Query

1. Upload a PDF document using the sidebar
2. Wait for ingestion to complete
3. Ask questions in the chat interface
4. Get grounded answers with citations

## Architecture

```
e:\ragprod/
├── config/
│   └── settings.py          # Configuration management
├── src/
│   ├── document/             # PDF parsing, chunking, table extraction
│   ├── stores/               # FAISS, Neo4j, SQLite, BM25
│   ├── retrieval/            # Query processing, hybrid retrieval, reranking
│   ├── generation/           # LLM engine, response building
│   └── pipeline/             # Ingestion and query orchestrators
├── api/
│   └── main.py               # FastAPI backend
├── frontend/
│   └── index.html            # Web UI
└── data/                     # Uploads, indexes, databases
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload and ingest PDF |
| `/query` | POST | RAG query with citations |
| `/documents` | GET | List ingested documents |
| `/health` | GET | System health check |

## Models Used

| Component | Model | VRAM |
|-----------|-------|------|
| Embeddings | bge-large-en-v1.5 | ~2GB |
| Reranker | bge-reranker-large | ~2GB |
| LLM | LLaMA-3-8B-Instruct (4-bit) | ~6GB |

## Running on Lightning AI

```bash
# Install dependencies
pip install -r requirements.txt

# Start API (Lightning AI will assign a URL)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Evaluation Metrics

- **Faithfulness**: Every claim traceable to context
- **Recall**: Relevant information retrieved
- **Citation Accuracy**: Correct section/page references
- **Failure Handling**: Proper "not found" responses

## License

MIT License
