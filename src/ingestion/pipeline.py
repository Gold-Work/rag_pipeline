"""Fonctions d'ingestion réutilisables.

Ce module expose deux points d'entrée :
- ingest_file(file_path)     : ingestion d'un seul fichier (utilisé par l'API upload)
- ingest_directory(data_path): ingestion batch d'un dossier (utilisé par run_ingestion.py)
"""
import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_documents
from src.ingestion.loader import (
    HASH_STORE,
    SUPPORTED_EXTENSIONS,
    compute_file_hash,
    load_documents,
    save_ingested_hashes,
)
from src.ingestion.parser import parse_documents
from src.query.cache import invalidate_cache
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def _load_hash_store() -> set[str]:
    if HASH_STORE.exists():
        with open(HASH_STORE, "r") as f:
            return set(json.load(f))
    return set()


def _load_single_file(file_path: Path, file_hash: str) -> List[Document]:
    """Charge un fichier unique et injecte les métadonnées standard."""
    ext = file_path.suffix.lower()

    if ext == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    elif ext in SUPPORTED_EXTENSIONS:
        loader = SUPPORTED_EXTENSIONS[ext](str(file_path))
    else:
        raise ValueError(f"Extension non supportée : {ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["file_type"] = ext
        doc.metadata["file_hash"] = file_hash

    return docs


def ingest_file(file_path: Path) -> int:
    """Ingère un fichier unique sans toucher aux autres fichiers de l'index.

    - Vérifie le hash avant ingestion (idempotent).
    - Sauvegarde le hash uniquement si l'embedding réussit.
    - Invalide le cache query si de nouveaux chunks sont ajoutés.

    Returns:
        Nombre de chunks ajoutés à ChromaDB.
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"🚀 Ingestion fichier : {file_path.name}")
    logger.info(f"{'=' * 50}")

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    ingested_hashes = _load_hash_store()
    file_hash = compute_file_hash(file_path)

    if file_hash in ingested_hashes:
        logger.info(f"⏭️ Déjà ingéré (hash identique) : {file_path.name}")
        return 0

    # Étape 1 — Chargement
    logger.info(f"📂 Chargement : {file_path.name}")
    docs = _load_single_file(file_path, file_hash)
    if not docs:
        logger.warning(f"⚠️ Aucune page chargée depuis {file_path.name}")
        return 0
    logger.info(f"✅ {len(docs)} page(s) chargée(s)")

    # Étape 2 — Parsing
    logger.info("🔍 Parsing et nettoyage...")
    docs = parse_documents(docs)
    if not docs:
        logger.warning(f"⚠️ Aucun contenu utilisable après parsing : {file_path.name}")
        return 0
    logger.info(f"✅ {len(docs)} document(s) après nettoyage")

    # Étape 3 — Chunking
    logger.info("✂️  Chunking...")
    chunks = chunk_documents(docs)
    if not chunks:
        logger.warning(f"⚠️ Aucun chunk généré : {file_path.name}")
        return 0
    logger.info(f"✅ {len(chunks)} chunks créés")

    # Étape 4 — Embedding (embed_documents gère les doublons par chunk_id)
    logger.info("🧠 Embedding et indexation ChromaDB...")
    added = embed_documents(chunks)

    # Hash sauvegardé seulement après un embedding sans exception
    ingested_hashes.add(file_hash)
    save_ingested_hashes(ingested_hashes)

    logger.info(f"{'=' * 50}")
    logger.info(f"✅ Ingestion terminée : {file_path.name} — {added} chunks ajoutés")
    logger.info(f"{'=' * 50}")

    return added


def ingest_directory(data_path: str) -> int:
    """Ingère tous les nouveaux fichiers d'un dossier.

    Délègue la détection des nouveaux fichiers à load_documents (hash store).
    Utilisé uniquement par run_ingestion.py.

    Returns:
        Nombre total de chunks ajoutés à ChromaDB.
    """
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE PIPELINE INGESTION BATCH")
    logger.info("=" * 60)

    documents, ingested_hashes = load_documents(data_path)
    if not documents:
        logger.info("ℹ️ Aucun nouveau document à ingérer.")
        return 0
    logger.info(f"✅ {len(documents)} page(s) chargée(s)")

    documents = parse_documents(documents)
    logger.info(f"✅ {len(documents)} document(s) après nettoyage")

    chunks = chunk_documents(documents)
    logger.info(f"✅ {len(chunks)} chunks créés")

    added = embed_documents(chunks)

    # Hash store écrit après embedding réussi
    save_ingested_hashes(ingested_hashes)

    invalidate_cache()

    logger.info("=" * 60)
    logger.info(f"✅ PIPELINE BATCH TERMINÉE — {added} chunks ajoutés")
    logger.info("=" * 60)

    return added
