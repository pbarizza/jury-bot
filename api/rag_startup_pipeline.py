# api/rag_startup_pipeline.py

import os
import re
import unicodedata
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from groq import Groq  # <-- New import

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Load Environment Variables
# -------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "startup-rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "./api/startups_data.csv")
PITCH_DECK_DIR = os.getenv("PITCH_DECK_DIR", "./api/pitch_decks")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required env vars
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY is missing")
    raise EnvironmentError("PINECONE_API_KEY must be set")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is missing")
    raise EnvironmentError("GROQ_API_KEY must be set")

# -------------------------------
# Initialize Services
# -------------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("✅ Connected to Pinecone")
except Exception as e:
    logger.error(f"❌ Failed to connect to Pinecone: {e}")
    raise

# Lazy-load embedding model
model = None

def get_embedding_model():
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

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Connected to Groq")
except Exception as e:
    logger.error(f"❌ Failed to initialize Groq client: {e}")
    raise

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Jury-Bot RAG API",
    description="Vector search + LLM generation for startup evaluation",
    version="1.0.0"
)

# -------------------------------
# Utility: Sanitize IDs
# -------------------------------
def sanitize_id(text: str) -> str:
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
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def read_pitch_deck(file_name):
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
# Ingest Data
# -------------------------------
def ingest_data():
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"✅ Loaded {len(df)} startups from {CSV_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise RuntimeError(f"CSV load failed: {e}")

    df = df.fillna("")

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
    try:
        indexes = pc.list_indexes().names()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "index_exists": INDEX_NAME in indexes,
            "model_loaded": model is not None,
            "groq_connected": groq_client is not None
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    model: Optional[str] = "llama3-8b-8192"  # Allow model override

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        top_k = max(1, min(request.top_k, 10))
        llm_model = request.model

        # Validate LLM model (optional safety check)
        allowed_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        if llm_model not in allowed_models:
            raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {allowed_models}")

        # Get query embedding
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(question).tolist()

        # Query Pinecone
        index = pc.Index(INDEX_NAME)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        # Collect contexts and metadata
        contexts = []
        matches = []
        for match in results['matches']:
            meta = match['metadata']
            full_text = meta.get('full_text', '')
            contexts.append(full_text)

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
                "full_context": full_text
            })

        # Build prompt for Groq
        context_str = "\n\n---\n\n".join(contexts[:3])  # Use top 3
        prompt = f"""
        You are an expert startup evaluator. Answer the question using only the context below.

        Context:
        {context_str}

        Question:
        {question}

        Answer concisely and professionally:
        """

        # Call Groq LLM
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=llm_model,
            temperature=0.3,
            max_tokens=512,
        )
        generated_answer = chat_completion.choices[0].message.content.strip()

        return {
            "question": question,
            "generated_answer": generated_answer,
            "retrieved_results": matches,
            "context_used": context_str
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
