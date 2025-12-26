# Running on Lightning AI

This guide explains how to run the Enterprise RAG System on Lightning AI with GPU support.

## Why Lightning AI?

The full model stack requires ~10-12GB VRAM:
- Embeddings (bge-large-en-v1.5): ~2GB
- Reranker (bge-reranker-large): ~2GB  
- LLM (LLaMA-3-8B-Instruct 4-bit): ~6GB

Your RTX 3050 (4GB) cannot fit all models simultaneously. Lightning AI provides free GPU access with 16-24GB VRAM.

## Step 1: Create Lightning AI Account

1. Go to [lightning.ai](https://lightning.ai)
2. Sign up for a free account
3. Create a new **Studio**

## Step 2: Set Up the Studio

1. Click **"New Studio"**
2. Select **GPU** machine:
   - Recommended: **L4** (24GB VRAM) or **T4** (16GB VRAM)
   - Free tier includes GPU hours
3. Wait for the Studio to start

## Step 3: Upload Project

### Option A: Clone from Local (Recommended)

1. Zip your project:
```bash
# On your local machine
cd e:\ragprod
# Create zip excluding data folders
powershell Compress-Archive -Path * -DestinationPath ragprod.zip -Force
```

2. In Lightning AI Studio:
   - Click the file browser
   - Upload `ragprod.zip`
   - Extract: `unzip ragprod.zip -d ragprod`

### Option B: Use Git

If you push to GitHub:
```bash
git clone https://github.com/YOUR_USERNAME/ragprod.git
cd ragprod
```

## Step 4: Install Dependencies

Open the terminal in Lightning AI and run:

```bash
cd ragprod

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Step 5: Configure Environment

Create a `.env` file for production settings:

```bash
cat > .env << 'EOF'
# Model configuration
RAG_MODEL_DEVICE=cuda
RAG_MODEL_LLM_QUANTIZATION=4bit

# Store configuration (Neo4j optional)
RAG_STORE_NEO4J_URI=bolt://localhost:7687
RAG_STORE_NEO4J_USER=neo4j
RAG_STORE_NEO4J_PASSWORD=password

# API configuration
RAG_API_HOST=0.0.0.0
RAG_API_PORT=8000
RAG_DEBUG=false
EOF
```

## Step 6: Start the Server

```bash
cd ragprod

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Make it Accessible

Lightning AI provides a public URL for your app:

1. Look for **"Open Port"** or **"Expose"** button in the interface
2. Expose port **8000**
3. You'll get a URL like: `https://xxxxx.lightning.ai`

## Step 7: Access the Frontend

Open the provided Lightning AI URL in your browser:
- Upload PDFs
- Query your documents
- Get grounded answers with citations

## Optional: Start Neo4j (for Knowledge Graph)

If you want the full knowledge graph functionality:

```bash
# Install Docker in the Studio (if not available)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5

# Wait for Neo4j to start
sleep 30

# Verify connection
curl http://localhost:7474
```

Without Neo4j, the system falls back to in-memory mode which still works for basic retrieval.

## Monitoring GPU Usage

Check GPU memory during operation:

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Or get current usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

Expected usage during query:
- Idle: ~0.5GB
- After loading embeddings: ~2.5GB
- After loading reranker: ~4.5GB
- After loading LLM: ~10-11GB

## Running Notebooks

For interactive development:

```bash
# Install Jupyter
pip install jupyterlab

# Start Jupyter (Lightning AI auto-exposes this)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Open notebooks/01_ingestion_demo.ipynb
```

## Cost Optimization Tips

1. **Stop when not using**: Studios stop automatically after inactivity
2. **Use smaller models locally**: For development, test with:
   - `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
   - Skip reranker for testing
3. **Batch your work**: Upload multiple PDFs, run all queries, then stop

## Troubleshooting

### Out of Memory
```python
# In config/settings.py, reduce batch sizes:
# Change in VectorStore
batch_size = 16  # Instead of 32

# Use 8-bit instead of 4-bit if GPU supports it
RAG_MODEL_LLM_QUANTIZATION=8bit
```

### CUDA Not Available
```bash
# Verify GPU is accessible
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

### Neo4j Connection Failed
The system automatically falls back to memory mode. This is fine for testing.

### Model Download Slow
First run downloads models (~5-10GB). This is one-time only.

```bash
# Pre-download models
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-large-en-v1.5')
CrossEncoder('BAAI/bge-reranker-large')
print('Models downloaded!')
"
```

## CPU-Only Mode (for Local Testing)

If you want to test locally on your RTX 3050 (limited):

```bash
# Set CPU mode in .env
RAG_MODEL_DEVICE=cpu

# Use smaller embedding model
# Edit config/settings.py:
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
```

This allows testing without GPU, but will be slow for LLM generation.
