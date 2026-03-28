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
        separators=["\n\n", "\n", ".", " "],  # pas besoin du séparateur vide ""
        length_function=len,
    )

    # Découpage en chunks
    chunks = splitter.split_documents(documents)

    # Enrichir les métadonnées avec un ID unique
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk.page_content.encode(), usedforsecurity=False).hexdigest()[:12]
        source = chunk.metadata.get("source_file", "unknown")
        chunk.metadata["chunk_id"] = f"{source}_{i}_{content_hash}"
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
