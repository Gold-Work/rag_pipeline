import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger("chunker")


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe les documents en chunks avec un ID unique pour chaque chunk.

    Args:
        documents (List[Document]): Liste de documents nettoyés

    Returns:
        List[Document]: Liste de chunks avec métadonnées enrichies
    """
    # Récupérer la configuration depuis config.yaml
    config = get_config()
    chunk_size = int(config["retrieval"]["chunk_size"])
    chunk_overlap = int(config["retrieval"]["chunk_overlap"])

    logger.info(f"✂️  Chunking — taille: {chunk_size}, overlap: {chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],  # ". " évite les coupures sur abréviations (Dr., Fig.)
        length_function=len,
    )

    # Découpage en chunks
    chunks = splitter.split_documents(documents)

    # Enrichir les métadonnées avec un ID unique
    # L'index i est local au fichier source pour rester stable entre sessions
    local_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        i = local_index.get(source, 0)
        local_index[source] = i + 1
        content_sig = f"{source}|{i}|{chunk.page_content}"
        content_hash = hashlib.md5(content_sig.encode(), usedforsecurity=False).hexdigest()[:16]
        chunk.metadata["chunk_id"] = f"{source}_{content_hash}"
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info(f"✅ {len(chunks)} chunks créés depuis {len(documents)} documents")

    # Statistiques par fichier
    stats: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "inconnu")
        stats[source] = stats.get(source, 0) + 1

    for source, count in stats.items():
        logger.info(f"   📄 {source} → {count} chunks")

    return chunks
