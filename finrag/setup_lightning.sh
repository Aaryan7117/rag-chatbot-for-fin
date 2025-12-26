#!/bin/bash
# ================================================
# Lightning AI Quick Start Script
# Run this in your Lightning AI Studio terminal
# ================================================

echo "🚀 Enterprise RAG System - Lightning AI Setup"
echo "=============================================="

# Check if GPU is available
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ PyTorch not installed or GPU check failed"
    echo "Installing PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "📥 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Create data directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/uploads data/processed data/faiss_index data/bm25_index data/tables

# Pre-download models (optional but recommended)
echo ""
echo "🤖 Pre-downloading ML models (this may take 5-10 minutes)..."
python << 'EOF'
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-large-en-v1.5')
print("✓ Embedding model ready")

print("Loading reranker model...")
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-large')
print("✓ Reranker model ready")

print("")
print("✅ All models downloaded successfully!")
print("Note: LLM will be downloaded on first query (additional ~5GB)")
EOF

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then expose port 8000 in Lightning AI to access the UI"
echo "=============================================="
