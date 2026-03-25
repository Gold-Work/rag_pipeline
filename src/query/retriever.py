import os
import re
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.query.rewriter import rewrite_query
from rank_bm25 import BM25Okapi
import chromadb

logger = get_logger("retriever")
config = get_config()

# Paramètres
top_k_retrieval = config["retrieval"]["top_k_retrieval"]
top_k_rerank = config["retrieval"]["top_k_rerank"]
embedding_model = config["embedding"]["model"]

_vector_store = None
_bm25 = None
_bm25_docs = None

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=os.getenv("OPENAI_API_KEY"))
        client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
        _vector_store = Chroma(
            client=client,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
            embedding_function=embeddings,
        )
        logger.info("✅ Vector store initialisé (lazy singleton)")
    return _vector_store

def get_bm25():
    global _bm25, _bm25_docs
    if _bm25 is None:
        vector_store = get_vector_store()
        data = vector_store.get(include=["documents", "metadatas"])
        if not data["documents"]:
            logger.warning("⚠️ BM25 : aucun document trouvé")
            return None, []
        _bm25_docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(data["documents"], data["metadatas"])]
        tokenized = [tokenize(d.page_content) for d in _bm25_docs]
        _bm25 = BM25Okapi(tokenized)
        logger.info(f"✅ BM25 initialisé sur {len(_bm25_docs)} documents")
    return _bm25, _bm25_docs

def vector_search(query: str, k: int) -> list[tuple[Document, float]]:
    return get_vector_store().similarity_search_with_score(query, k=k) or []

def bm25_search(query: str, k: int) -> list[tuple[Document, float]]:
    bm25, docs = get_bm25()
    if not bm25: return []
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], scores[i]) for i in top_idx if scores[i] > 0]

def reciprocal_rank_fusion(vector_results, bm25_results, k=60) -> list[Document]:
    scores = {}
    doc_map = {}
    for rank, (doc, _) in enumerate(vector_results):
        key = doc.metadata.get("chunk_id", hash(doc.page_content))
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc
    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.metadata.get("chunk_id", hash(doc.page_content))
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc
    return [doc_map[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def retrieve(query: str, k: int = 5) -> list[Document]:
    logger.info(f"🔍 Query : '{query}'")
    queries = rewrite_query(query)
    all_vector_results, all_bm25_results = [], []

    for q in queries:
        all_vector_results.extend(vector_search(q, k=top_k_retrieval))
        all_bm25_results.extend(bm25_search(q, k=top_k_retrieval))

    # Déduplication
    seen = set()
    all_vector_results = [r for r in all_vector_results if not (r[0].page_content in seen or seen.add(r[0].page_content))]
    seen = set()
    all_bm25_results = [r for r in all_bm25_results if not (r[0].page_content in seen or seen.add(r[0].page_content))]

    if not all_vector_results and not all_bm25_results:
        logger.warning("⚠️ Aucun résultat trouvé")
        return []

    fused = reciprocal_rank_fusion(all_vector_results, all_bm25_results)
    final = fused[:k]

    logger.info(f"📊 {len(final)} chunks retenus après fusion RRF")
    return final