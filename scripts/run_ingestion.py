import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.utils.logger import get_logger
from src.ingestion.loader import load_documents, save_ingested_hashes
from src.ingestion.parser import parse_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_documents
from src.query.cache import invalidate_cache

logger = get_logger("run_ingestion")


def run_ingestion():
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE PIPELINE INGESTION")
    logger.info("=" * 60)

    data_path = os.getenv("DATA_RAW_PATH", "./data/raw")

    # ÉTAPE 1 — Chargement
    logger.info("📂 ÉTAPE 1 : Chargement des documents...")
    documents, ingested_hashes = load_documents(data_path)
    if not documents:
        logger.error("❌ Aucun document chargé. Vérifiez le dossier data/raw/")
        return
    logger.info(f"✅ {len(documents)} pages/sections chargées")

    # ÉTAPE 2 — Parsing & Cleaning
    logger.info("🔍 ÉTAPE 2 : Parsing et nettoyage...")
    documents = parse_documents(documents)
    logger.info(f"✅ {len(documents)} documents après nettoyage")

    # ÉTAPE 3 — Chunking
    logger.info("✂️  ÉTAPE 3 : Chunking...")
    chunks = chunk_documents(documents)
    logger.info(f"✅ {len(chunks)} chunks créés")

    # ÉTAPE 4 — Embedding & Vector Store
    logger.info("🧠 ÉTAPE 4 : Embedding et indexation...")
    added = embed_documents(chunks)

    # Hash store écrit APRÈS embedding réussi pour éviter la déconnexion hash/ChromaDB
    save_ingested_hashes(ingested_hashes)

    # Invalider le cache query pour que les nouvelles requêtes voient les nouveaux chunks
    invalidate_cache()

    # RÉSUMÉ FINAL
    logger.info("=" * 60)
    logger.info("✅ PIPELINE INGESTION TERMINÉE")
    logger.info(f"📊 Chunks ajoutés à ChromaDB : {added}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
