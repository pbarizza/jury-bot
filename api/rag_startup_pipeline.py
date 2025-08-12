# api/rag_startup_pipeline.py

import os
import re
import unicodedata
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import logging

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Load Environment Variables
# -------------------------------
from dotenv import load_dotenv  # Optional: only if using .env locally

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "startup-rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "./api/startups_data.csv")
PITCH_DECK_DIR = os.getenv("PITCH_DECK_DIR", "./api/pitch_decks")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # For future LLM use

# Validate required env vars (Pinecone key is essential)
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY is missing")
    raise EnvironmentError("PINECONE_API_KEY must be set")

# -------------------------------
# Initialize Services
# -------------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("✅ Connected to Pinecone")
except Exception as e:
    logger.error(f"❌ Failed to connect to Pinecone: {e}")
    raise

# We'll lazy-load the model to avoid startup issues
model = None

def get_embedding_model():
    """Lazy load the embedding model to handle cold starts"""
    global model
    if model is None:
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    return model

DIMENSION = 1024  # Matches BAAI/bge-large-en-v1.5

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Jury-Bot RAG API",
    description="Vector search backend for startup evaluation",
    version="1.0.0"
)

# -------------------------------
# Utility: Sanitize IDs (ASCII-only)
# -------------------------------
def sanitize_id(text: str) -> str:
    """Convert any string into a safe, ASCII-only ID"""
    if not isinstance(text, str) or text.strip() == "" or pd.isna(text):
        return "unknown"
    text = str(text).strip()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-zA-Z0-9 _\-]+', '', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = re.sub(r'[-]+', '-', text)
    return text.strip('_-')[:200]

# -------------------------------
# Text & File Utilities
# -------------------------------
def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def read_pitch_deck(file_name):
    """Safely read pitch deck with fallbacks"""
    if pd.isna(file_name) or not isinstance(file_name, str):
        return "No pitch deck provided."

    file_name = file_name.strip()
    if not file_name.lower().endswith('.txt'):
        file_name += '.txt'

    file_path = os.path.join(PITCH_DECK_DIR, file_name)

    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return clean_text(f.read())
        else:
            logger.warning(f"Pitch deck not found: {file_path}")
            return "Pitch deck file not found."
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return "Error reading pitch deck."

# -------------------------------
# Prepare Document for Embedding
# -------------------------------
def prepare_startup_document(row):
    """Combine key fields and pitch deck into a searchable text"""
    pitch_text = read_pitch_deck(row.get('Pitch Deck File', ''))

    combined_text = f"""
    Startup Name: {row.get('Startup Name', 'N/A')}
    Domain: {row.get('Domain', 'N/A')}, Subdomain: {row.get('Subdomain', 'N/A')}
    Stage: {row.get('Stage', 'N/A')}, Team Size: {row.get('Team Size', 'N/A')}, Monthly Revenue: {row.get('Monthly Revenue', 'N/A')}
    Funding: {row.get('Has Funding', 'N/A')} (Amount: {row.get('Funding Amount', 'N/A')}, Stage: {row.get('Funding Stage', 'N/A')})
    Description: {row.get('Description', 'N/A')}
    Problem & Solution: {row.get('Problem/Solution', 'N/A')}
    Vision: {row.get('Vision', 'N/A')}
    Business Model: {row.get('Business Model', 'N/A')}
    Technologies: {row.get('Technologies', 'N/A')}
    Advantages: {row.get('Advantages', 'N/A')}
    Founding Members: {row.get('Founding Members', 'N/A')} (Experience: {row.get('Experience', 'N/A')} years)
    Pitch Deck Summary: {pitch_text}
    """
    return clean_text(combined_text)

# -------------------------------
# Ingest Data (Optional: Run via CLI)
# -------------------------------
def ingest_data():
    """Load CSV and upsert into Pinecone"""
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(INDEX_NAME)

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"✅ Loaded {len(df)} startups from {CSV_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise RuntimeError(f"CSV load failed: {e}")

    df = df.fillna("")  # Clean NaNs

    batch_size = 32
    for i in range(0, len(df), batch_size):
        i_end = min(i + batch_size, len(df))
        batch = df.iloc[i:i_end]

        ids = []
        texts = []
        metadatas = []

        for _, row in batch.iterrows():
            startup_name = row.get('Startup Name', 'Unknown')
            source = row.get('Source', 'Unknown')
            safe_startup = sanitize_id(startup_name)
            safe_source = sanitize_id(source)
            doc_id = f"startup-{safe_startup}-{safe_source}"[:512]

            text = prepare_startup_document(row)

            metadata = {k: "" if pd.isna(v) or v is None else str(v) for k, v in row.items() if pd.notna(k)}
            metadata['full_text'] = text

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        # Generate embeddings
        embeds = get_embedding_model().encode(texts).tolist()
        to_upsert = list(zip(ids, embeds, metadatas))
        index.upsert(vectors=to_upsert)
        logger.info(f"✅ Upserted batch {i} to {i_end}")

    logger.info("🎉 Data ingestion complete!")

# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/")
def home():
    return {
        "status": "alive",
        "service": "Jury-Bot RAG API",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        indexes = pc.list_indexes().names()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "index_exists": INDEX_NAME in indexes,
            "model_loaded": model is not None
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        top_k = max(1, min(request.top_k, 10))

        # Lazy load model
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(question).tolist()

        index = pc.Index(INDEX_NAME)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        matches = []
        for match in results['matches']:
            meta = match['metadata']
            matches.append({
                "id": match['id'],
                "score": match['score'],
                "startup_name": meta.get('Startup Name', 'Unknown'),
                "domain": meta.get('Domain', 'N/A'),
                "subdomain": meta.get('Subdomain', 'N/A'),
                "stage": meta.get('Stage', 'N/A'),
                "funding_stage": meta.get('Funding Stage', 'N/A'),
                "team_size": meta.get('Team Size', 'N/A'),
                "description": meta.get('Description', '')[:300] + "...",
                "full_context": meta.get('full_text', '')
            })

        return {
            "question": question,
            "results": matches
        }

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# -------------------------------
# CLI: Run ingest or serve
# -------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "ingest":
            ingest_data()
        else:
            print("Usage: python rag_startup_pipeline.py ingest")
    else:
        print("Run via uvicorn (see render.yaml)")
