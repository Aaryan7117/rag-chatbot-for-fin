"""
FastAPI Backend for Enterprise RAG System
==========================================

Provides REST API for:
- PDF upload and ingestion
- RAG queries with citations
- Document listing
- Health checks

Works locally and on Lightning AI.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.pipeline.ingestion import IngestionPipeline, IngestionResult
from src.pipeline.query import QueryPipeline, RAGResponse


# ============== Pydantic Models ==============

class QueryRequest(BaseModel):
    """Request body for RAG query."""
    query: str = Field(..., min_length=1, description="User question")
    doc_id: Optional[str] = Field(None, description="Optional document filter")
    top_k: int = Field(5, ge=1, le=10, description="Number of sources")


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    has_answer: bool
    confidence: float
    citations: List[dict]
    tables: List[str]
    source_count: int
    processing_time_ms: float


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    doc_id: str
    filename: str
    title: Optional[str]
    status: str
    chunks: int
    tables: int
    ingested_at: str


class UploadResponse(BaseModel):
    """Response from document upload."""
    success: bool
    doc_id: str
    filename: str
    message: str
    chunks_created: int = 0
    tables_extracted: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    stores: dict


# ============== Application Setup ==============

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Enterprise RAG System",
        description="High-accuracy RAG for SEC 10-K filings and financial documents",
        version="1.0.0"
    )
    
    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize settings
    settings.initialize()
    
    # Store instances (lazy-loaded)
    app.state.ingestion_pipeline = None
    app.state.query_pipeline = None
    app.state.documents = {}  # doc_id -> DocumentInfo
    
    return app


app = create_app()


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get or create ingestion pipeline."""
    if app.state.ingestion_pipeline is None:
        app.state.ingestion_pipeline = IngestionPipeline()
    return app.state.ingestion_pipeline


def get_query_pipeline() -> QueryPipeline:
    """Get or create query pipeline."""
    if app.state.query_pipeline is None:
        app.state.query_pipeline = QueryPipeline()
    return app.state.query_pipeline


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all stores.
    """
    stores = {
        "vector_store": "unknown",
        "bm25_index": "unknown",
        "knowledge_graph": "unknown",
        "table_store": "unknown"
    }
    
    try:
        pipeline = get_ingestion_pipeline()
        stores["vector_store"] = f"{pipeline.vector_store.count} vectors"
        stores["bm25_index"] = f"{pipeline.bm25_index.count} chunks"
        stores["knowledge_graph"] = "connected" if pipeline.knowledge_graph.is_connected else "memory mode"
        stores["table_store"] = f"{pipeline.table_store.count} tables"
    except Exception as e:
        stores["error"] = str(e)
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        stores=stores
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and ingest a PDF document.
    
    The document is:
    1. Saved to uploads directory
    2. Processed through ingestion pipeline
    3. Added to all stores
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Save file
    upload_path = settings.paths.uploads_dir / file.filename
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Run ingestion
    try:
        pipeline = get_ingestion_pipeline()
        result = pipeline.ingest(upload_path)
        
        if result.success:
            # Store document info
            app.state.documents[result.doc_id] = DocumentInfo(
                doc_id=result.doc_id,
                filename=result.filename,
                title=result.title,
                status="ready",
                chunks=result.chunks_created,
                tables=result.tables_extracted,
                ingested_at=datetime.now().isoformat()
            )
            
            return UploadResponse(
                success=True,
                doc_id=result.doc_id,
                filename=file.filename,
                message=f"Document ingested successfully",
                chunks_created=result.chunks_created,
                tables_extracted=result.tables_extracted
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {', '.join(result.errors)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion error: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system.
    
    Returns grounded answer with citations.
    """
    try:
        pipeline = get_query_pipeline()
        
        response = pipeline.query(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
            verbose=True
        )
        
        return QueryResponse(
            answer=response.answer,
            has_answer=response.has_answer,
            confidence=response.confidence,
            citations=response.citations,
            tables=[t.get("table_id", "") for t in response.tables],
            source_count=len(response.source_chunks),
            processing_time_ms=response.processing_time_ms
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    return list(app.state.documents.values())


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """Get information about a specific document."""
    if doc_id not in app.state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found"
        )
    return app.state.documents[doc_id]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document (metadata only, stores not cleared)."""
    if doc_id not in app.state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found"
        )
    
    del app.state.documents[doc_id]
    return {"message": f"Document {doc_id} removed"}


# ============== Static Frontend ==============

# Serve frontend if available
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML."""
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    # Serve static files
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ============== Run Configuration ==============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
