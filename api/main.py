# api/main.py
from fastapi import FastAPI
import duckdb
import os
from contextlib import contextmanager
from typing import Dict
import json

from .generator import generate_single_synthetic_startup
from kafka_producer import send_startup_event

app = FastAPI(title="Startup Generator API")

# DuckDB setup
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/data/startups.duckdb")

@contextmanager
def get_db_connection():
    conn = duckdb.connect(DUCKDB_PATH)
    try:
        yield conn
    finally:
        conn.close()

@app.on_event("startup")
def setup_database():
    with get_db_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS startups (
            id INTEGER PRIMARY KEY,
            data JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

@app.post("/generate-startup", response_model=Dict)
def generate_startup():
    try:
        # 1. Generate synthetic startup
        startup_data = generate_single_synthetic_startup()

        # 2. Save to DuckDB
        with get_db_connection() as conn:
            # Get next ID
            result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM startups").fetchone()
            startup_id = result[0]

            conn.execute(
                "INSERT INTO startups (id, data) VALUES (?, ?)",
                [startup_id, json.dumps(startup_data)]
            )

        # 3. Send Kafka event
        send_startup_event({"id": startup_id, **startup_data})

        # 4. Return full startup
        return {"id": startup_id, "startup": startup_data}

    except Exception as e:
        print(f"Error generating startup: {e}")
        raise
