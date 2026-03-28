import re
from typing import List
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger("parser")

# Regex compilées pour performance
FORM_FEED = re.compile(r"\x0c")  # form feed PDF → frontière de paragraphe
CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")  # contrôles ASCII (\x0c géré séparément)
SOFT_HYPHEN = re.compile(r"\u00AD")  # soft hyphen PDF → supprimé
NBSP = re.compile(r"\u00A0")  # non-breaking space → espace normal
UNICODE_INVISIBLES = re.compile(r"[\u200B-\u200F\u2028\u2029\u2060-\u206F\uFEFF]")
TABS = re.compile(r"\t+")  # tabulations (tableaux PDF) → espace
MULTI_SPACES = re.compile(r" +")
MULTI_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Nettoyage avancé du texte, incluant caractères invisibles Unicode."""
    text = FORM_FEED.sub("\n\n", text)  # form feed → saut de paragraphe (avant CTRL_CHARS)
    text = CTRL_CHARS.sub("", text)  # autres contrôles ASCII
    text = SOFT_HYPHEN.sub("", text)  # soft hyphen → supprimé
    text = NBSP.sub(" ", text)  # non-breaking space → espace (évite fusion de mots)
    text = UNICODE_INVISIBLES.sub("", text)  # autres invisibles Unicode
    text = TABS.sub(" ", text)  # tabulations → espace
    text = MULTI_SPACES.sub(" ", text)  # espaces multiples → un
    text = MULTI_LINES.sub("\n\n", text)  # 3+ sauts de ligne → 2 max
    return text.strip()


def parse_documents(documents: List[Document], min_length: int = 20) -> List[Document]:
    """
    Nettoie et filtre les documents trop courts.

    Args:
        documents (List[Document]): Liste de documents LangChain
        min_length (int): Longueur minimale du texte pour être conservé

    Returns:
        List[Document]: Liste des documents nettoyés
    """
    logger.info(f"🔍 Parsing de {len(documents)} documents...")

    cleaned = []
    skipped = 0

    for doc in documents:
        original_len = len(doc.page_content)
        cleaned_text = clean_text(doc.page_content)

        if len(cleaned_text) < min_length:
            logger.warning(
                f"⚠️  Page trop courte ignorée ({len(cleaned_text)} chars) — {doc.metadata.get('source_file', '?')}"
            )
            skipped += 1
            continue

        new_doc = Document(page_content=cleaned_text, metadata=dict(doc.metadata))
        cleaned.append(new_doc)

        logger.debug(f"📝 {doc.metadata.get('source_file', '?')} | {original_len} → {len(cleaned_text)} chars")

    logger.info(f"✅ Parsing terminé : {len(cleaned)} documents conservés, {skipped} ignorés")
    return cleaned
