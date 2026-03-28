import os
import time
import uuid
import openai
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.query.cache import invalidate_cache
import chromadb
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("embedder")
config = get_config()


def _ensure_chunk_id(chunk: Document) -> None:
    """Assign a stable chunk_id if the chunk doesn't already have one.

    The normal path (chunk_documents → embed_documents) always sets chunk_id
    in the chunker.  This guard catches any document that bypasses chunking
    (e.g., direct calls to embed_documents in tests or scripts).

    The ID is a UUID5 derived from source + content, so it is deterministic
    and collision-resistant.
    """
    if chunk.metadata.get("chunk_id"):
        return
    content_sig = f"{chunk.metadata.get('source_file', '')}|{chunk.page_content}"
    chunk.metadata["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_sig))
    logger.warning(f"⚠️ chunk_id manquant — UUID généré pour : {chunk.metadata.get('source_file', '?')}")


def get_embeddings():
    return OpenAIEmbeddings(
        model=config["embedding"]["model"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_vector_store(embeddings) -> Chroma:
    client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
    return Chroma(
        client=client,
        collection_name=config["embedding"]["collection_name"],
        embedding_function=embeddings,
    )


def embed_with_retry(vector_store, batch, ids, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            vector_store.add_documents(documents=batch, ids=ids)
            return True
        except openai.OpenAIError as e:
            wait = 2**attempt  # 2s, 4s, 8s
            if attempt < max_retries:
                logger.warning(f"⚠️  Tentative {attempt}/{max_retries} échouée : {e} — retry dans {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"❌ Échec définitif après {max_retries} tentatives : {e}")
                return False


def embed_documents(chunks: List[Document]) -> int:
    logger.info(f"🚀 Début embedding de {len(chunks)} chunks...")

    # Guarantee every chunk has a stable chunk_id before touching ChromaDB.
    for chunk in chunks:
        _ensure_chunk_id(chunk)

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    # Récupérer les IDs déjà présents dans ChromaDB
    existing = vector_store.get()
    existing_ids = set(existing["ids"]) if existing["ids"] else set()
    logger.info(f"📦 Chunks déjà dans ChromaDB : {len(existing_ids)}")

    # Filtrer les chunks déjà indexés
    new_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_id") not in existing_ids]

    if not new_chunks:
        logger.info("✅ Aucun nouveau chunk à indexer.")
        return 0

    logger.info(f"📤 {len(new_chunks)} nouveaux chunks à embedder...")

    batch_size = config["embedding"]["batch_size"]
    total_batches = (len(new_chunks) + batch_size - 1) // batch_size
    total_success = 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        ids = [chunk.metadata["chunk_id"] for chunk in batch]
        batch_num = (i // batch_size) + 1

        success = embed_with_retry(vector_store, batch, ids)
        if success:
            total_success += len(batch)
            logger.info(f"✅ Batch {batch_num}/{total_batches} — {len(batch)} chunks indexés")
        else:
            logger.error(f"❌ Batch {batch_num}/{total_batches} abandonné")

    logger.info(f"🎉 Embedding terminé — {total_success}/{len(new_chunks)} chunks ajoutés au vector store")

    if total_success > 0:
        invalidate_cache()

    return total_success
