from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger("reranker")

_cross_encoder = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("🔄 Chargement du Cross-Encoder...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("✅ Cross-Encoder chargé (lazy singleton)")
    return _cross_encoder

def rerank(query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
    if not documents:
        logger.warning("⚠️  Aucun document à reranker")
        return []

    logger.info(f"🎯 Reranking de {len(documents)} chunks...")

    model = get_cross_encoder()

    # Préparer les paires (query, chunk)
    pairs = [(query, doc.page_content) for doc in documents]

    # Scorer toutes les paires
    scores = model.predict(pairs)

    # Associer scores et documents
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Log des scores
    logger.info(f"📊 Scores après reranking :")
    for i, (doc, score) in enumerate(scored_docs[:top_k]):
        logger.info(f"   [{i+1}] score={score:.4f} | source={doc.metadata.get('source_file', '?')} | aperçu={doc.page_content[:80].strip()}...")

    return [doc for doc, _ in scored_docs[:top_k]]