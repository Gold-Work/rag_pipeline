from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.config import get_config
import hashlib
import json

logger = get_logger("loader")
config = get_config()

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,  # PyMuPDFLoader plus fiable que PyPDFLoader
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


# Calcul hash pour éviter ré-ingestion
def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


HASH_STORE = Path(config["paths"]["ingested_files"])
HASH_STORE.parent.mkdir(parents=True, exist_ok=True)


def load_documents(data_path: str) -> tuple[List[Document], set[str]]:
    """Charge les documents non encore ingérés.

    Returns:
        (documents, ingested_hashes) — le hash store n'est PAS écrit ici.
        Appeler save_ingested_hashes(ingested_hashes) après l'embedding réussi
        pour éviter de marquer des fichiers comme traités en cas d'échec.
    """
    if HASH_STORE.exists():
        with open(HASH_STORE, "r") as f:
            ingested_hashes = set(json.load(f))
    else:
        ingested_hashes = set()

    documents = []
    base = Path(data_path)
    files = list(base.rglob("*"))
    logger.info(f"📂 Dossier scanné (récursif) : {data_path}")

    for file in files:
        if not file.is_file():
            continue
        ext = file.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS and ext != ".txt":
            continue

        file_hash = compute_file_hash(file)
        if file_hash in ingested_hashes:
            logger.info(f"⏭️ Déjà traité (hash) : {file.name}")
            continue

        # Chemin relatif au dossier source pour éviter les collisions de noms
        relative_path = str(file.relative_to(base))

        try:
            if ext == ".txt":
                loader = TextLoader(str(file), encoding="utf-8", autodetect_encoding=True)
            else:
                loader = SUPPORTED_EXTENSIONS[ext](str(file))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = relative_path
                doc.metadata["file_type"] = ext
                doc.metadata["file_hash"] = file_hash

            documents.extend(docs)
            ingested_hashes.add(file_hash)
            logger.info(f"✅ Chargé : {relative_path} ({len(docs)} pages/sections)")

        except OSError as e:
            logger.error(f"❌ Erreur d'accès fichier {relative_path} : {e}")
        except (RuntimeError, ValueError, UnicodeDecodeError) as e:
            logger.error(f"❌ Erreur de parsing {relative_path} : {e}", exc_info=True)

    logger.info(f"📊 Total documents chargés : {len(documents)}")
    return documents, ingested_hashes


def save_ingested_hashes(hashes: set[str]) -> None:
    """Persiste le hash store. À appeler après embedding réussi."""
    with open(HASH_STORE, "w") as f:
        json.dump(list(hashes), f)
    logger.info(f"💾 Hash store sauvegardé — {len(hashes)} fichiers enregistrés")
