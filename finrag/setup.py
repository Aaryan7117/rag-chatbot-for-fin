"""
Lightning AI Quick Start Script (Windows/Python version)
Run this in your Lightning AI Studio or locally
"""

import subprocess
import sys
import os

def run_cmd(cmd, shell=True):
    """Run a command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=False)
    return result.returncode == 0

def main():
    print("🚀 Enterprise RAG System - Setup")
    print("=" * 50)
    
    # Check GPU
    print("\n📊 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ No GPU detected - will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")
        run_cmd(f"{sys.executable} -m pip install torch")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    run_cmd(f"{sys.executable} -m pip install -r requirements.txt")
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data ready")
    except Exception as e:
        print(f"⚠️ NLTK download failed: {e}")
    
    # Create directories
    print("\n📁 Creating directories...")
    dirs = [
        "data/uploads",
        "data/processed", 
        "data/faiss_index",
        "data/bm25_index",
        "data/tables"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created")
    
    # Pre-download models
    print("\n🤖 Pre-downloading ML models...")
    print("   This may take 5-10 minutes on first run...")
    
    try:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("   Loading embedding model (bge-large-en-v1.5)...")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("   ✅ Embedding model ready")
        
        print("   Loading reranker model (bge-reranker-large)...")
        from sentence_transformers import CrossEncoder
        CrossEncoder('BAAI/bge-reranker-large')
        print("   ✅ Reranker model ready")
        
    except Exception as e:
        print(f"⚠️ Model download failed: {e}")
        print("   Models will download on first use")
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nTo start the server:")
    print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("\nOr run directly:")
    print("  python api/main.py")
    print("=" * 50)
    
    # Ask to start server
    response = input("\n🚀 Start the server now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nStarting server on http://0.0.0.0:8000 ...")
        run_cmd(f"{sys.executable} -m uvicorn api.main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
