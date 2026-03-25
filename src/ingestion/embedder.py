import os
import time
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.utils.logger import get_logger
import chromadb
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("embedder")

def get_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_vector_store(embeddings) -> Chroma:
    persist_path = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    client = chromadb.PersistentClient(path=persist_path)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

def embed_with_retry(vector_store, batch, ids, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            vector_store.add_documents(documents=batch, ids=ids)
            return True
        except Exception as e:
            wait = 2 ** attempt  # 2s, 4s, 8s
            if attempt < max_retries:
                logger.warning(f"⚠️  Tentative {attempt}/{max_retries} échouée : {e} — retry dans {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"❌ Échec définitif après {max_retries} tentatives : {e}")
                return False

def embed_documents(chunks: List[Document]) -> Chroma:
    logger.info(f"🚀 Début embedding de {len(chunks)} chunks...")

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
        return vector_store

    logger.info(f"📤 {len(new_chunks)} nouveaux chunks à embedder...")

    # Embedding par batch de 100
    batch_size = 100
    total_batches = (len(new_chunks) + batch_size - 1) // batch_size
    total_success = 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        ids = [chunk.metadata["chunk_id"] for chunk in batch]
        batch_num = (i // batch_size) + 1

        success = embed_with_retry(vector_store, batch, ids)
        if success:
            total_success += len(batch)
            logger.info(f"✅ Batch {batch_num}/{total_batches} — {len(batch)} chunks indexés")
        else:
            logger.error(f"❌ Batch {batch_num}/{total_batches} abandonné")

    logger.info(f"🎉 Embedding terminé — {total_success}/{len(new_chunks)} chunks ajoutés au vector store")
    return vector_store