import hashlib
import os
import re
import threading
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.utils.observability import langfuse_span
from src.query.rewriter import rewrite_query
from src.query.cache import register_invalidation_hook
from rank_bm25 import BM25Okapi
import chromadb

logger = get_logger("retriever")
config = get_config()

# Paramètres
top_k_retrieval = config["retrieval"]["top_k_retrieval"]
top_k_rerank = config["retrieval"]["top_k_rerank"]
embedding_model = config["embedding"]["model"]

_vector_store = None
_vector_store_lock = threading.Lock()

_bm25 = None
_bm25_docs = None
_bm25_lock = threading.Lock()


def _reset_bm25() -> None:
    """Invalide le singleton BM25 — appelé automatiquement via cache hook."""
    global _bm25, _bm25_docs
    with _bm25_lock:
        _bm25 = None
        _bm25_docs = None
    logger.info("🔄 BM25 invalidé — sera reconstruit à la prochaine requête")


register_invalidation_hook(_reset_bm25)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def get_vector_store() -> Chroma:
    global _vector_store
    with _vector_store_lock:
        if _vector_store is None:
            embeddings = OpenAIEmbeddings(
                model=embedding_model, api_key=os.getenv("OPENAI_API_KEY")  # type: ignore[arg-type]
            )
            client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
            _vector_store = Chroma(
                client=client,
                collection_name=config["embedding"]["collection_name"],
                embedding_function=embeddings,
            )
            logger.info("✅ Vector store initialisé (lazy singleton)")
    return _vector_store


def get_bm25():
    global _bm25, _bm25_docs
    with _bm25_lock:
        if _bm25 is None:
            vector_store = get_vector_store()
            data = vector_store.get(include=["documents", "metadatas"])
            if not data["documents"]:
                logger.warning("⚠️ BM25 : aucun document trouvé")
                return None, []
            _bm25_docs = [
                Document(page_content=doc, metadata=meta) for doc, meta in zip(data["documents"], data["metadatas"])
            ]
            tokenized = [tokenize(d.page_content) for d in _bm25_docs]
            _bm25 = BM25Okapi(tokenized)
            logger.info(f"✅ BM25 initialisé sur {len(_bm25_docs)} documents")
    return _bm25, _bm25_docs


def vector_search(query: str, k: int) -> list[tuple[Document, float]]:
    return get_vector_store().similarity_search_with_score(query, k=k) or []


def bm25_search(query: str, k: int) -> list[tuple[Document, float]]:
    bm25, docs = get_bm25()
    if not bm25:
        return []
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], scores[i]) for i in top_idx if scores[i] > 0]


def _chunk_key(doc: Document) -> str:
    """Clé déterministe pour un chunk — chunk_id si présent, sinon MD5 du contenu."""
    return (
        doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode(), usedforsecurity=False).hexdigest()[:16]
    )


def reciprocal_rank_fusion(vector_results, bm25_results, k=60) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for rank, (doc, _) in enumerate(vector_results):
        key = _chunk_key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc
    for rank, (doc, _) in enumerate(bm25_results):
        key = _chunk_key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc
    return [(doc_map[key], score) for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve(query: str, k: int = 5) -> tuple[list[Document], list[str]]:
    logger.info(f"🔍 Query : '{query}'")

    with langfuse_span("query-rewriting", {"query": query}) as out:
        queries = rewrite_query(query)
        out["queries"] = queries
        out["n_variants"] = len(queries)

    all_vector_results: list = []
    all_bm25_results: list = []

    try:
        with langfuse_span("vector-search", {"queries": queries, "k": top_k_retrieval}) as out:
            for q in queries:
                all_vector_results.extend(vector_search(q, k=top_k_retrieval))
            out["results_count"] = len(all_vector_results)
    except Exception as e:
        logger.warning(f"⚠️ Vector search échoué, fallback BM25 seul : {e}")

    try:
        with langfuse_span("bm25-search", {"queries": queries, "k": top_k_retrieval}) as out:
            for q in queries:
                all_bm25_results.extend(bm25_search(q, k=top_k_retrieval))
            out["results_count"] = len(all_bm25_results)
    except Exception as e:
        logger.warning(f"⚠️ BM25 search échoué, fallback vector seul : {e}")

    # Déduplication
    seen: set[str] = set()
    deduped_vector: list = []
    for r in all_vector_results:
        if r[0].page_content not in seen:
            seen.add(r[0].page_content)
            deduped_vector.append(r)
    all_vector_results = deduped_vector

    seen = set()
    deduped_bm25: list = []
    for r in all_bm25_results:
        if r[0].page_content not in seen:
            seen.add(r[0].page_content)
            deduped_bm25.append(r)
    all_bm25_results = deduped_bm25

    if not all_vector_results and not all_bm25_results:
        logger.warning("⚠️ Aucun résultat trouvé")
        return [], []
    elif not all_vector_results:
        logger.info("ℹ️ Mode retrieval : BM25 seul (vector search indisponible)")
        fused = all_bm25_results  # déjà trié score décroissant
    elif not all_bm25_results:
        logger.info("ℹ️ Mode retrieval : vector seul (BM25 indisponible)")
        fused = all_vector_results  # déjà trié par pertinence décroissante
    else:
        logger.info("ℹ️ Mode retrieval : hybrid RRF")
        fused = reciprocal_rank_fusion(all_vector_results, all_bm25_results)

    min_score = float(config["retrieval"].get("min_retrieval_score", 0.0))
    if min_score > 0.0:
        before = len(fused)
        fused = [(doc, s) for doc, s in fused if s >= min_score]
        if len(fused) < before:
            logger.info(f"🔎 Score threshold {min_score} : {before - len(fused)} chunks filtrés")

    final = [doc for doc, _ in fused[:k]]
    chunk_ids = [doc.metadata.get("chunk_id", "") for doc in final]

    logger.info(f"📊 {len(final)} chunks retenus")
    return final, chunk_ids
