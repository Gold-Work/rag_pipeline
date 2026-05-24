"""Point d'entrée CLI pour l'ingestion batch.

Usage:
    python scripts/run_ingestion.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.pipeline import ingest_directory

if __name__ == "__main__":
    data_path = os.getenv("DATA_RAW_PATH", "./data/raw")
    ingest_directory(data_path)
