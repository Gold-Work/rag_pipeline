import threading
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.utils.observability import langfuse_span

logger = get_logger("reranker")
config = get_config()

_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is None:
            model_name = config["reranker"]["model"]
            logger.info("🔄 Chargement du Cross-Encoder...")
            _cross_encoder = CrossEncoder(model_name)
            logger.info("✅ Cross-Encoder chargé (lazy singleton)")
    return _cross_encoder


def rerank(query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
    if not documents:
        logger.warning("⚠️  Aucun document à reranker")
        return []

    logger.info(f"🎯 Reranking de {len(documents)} chunks...")

    with langfuse_span(
        "cross-encoder",
        {"query": query, "n_docs": len(documents), "top_k": top_k},
    ) as out:
        model = get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        logger.info("📊 Scores après reranking :")
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            logger.info(
                f"   [{i+1}] score={score:.4f} | "
                f"source={doc.metadata.get('source_file', '?')} | "
                f"aperçu={doc.page_content[:80].strip()}..."
            )

        result = [doc for doc, _ in scored_docs[:top_k]]
        out["top_scores"] = [round(float(s), 4) for _, s in scored_docs[:top_k]]

    return result
