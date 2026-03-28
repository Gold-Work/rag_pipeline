import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.config import get_config
import hashlib
import json

logger = get_logger("loader")
config = get_config()

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,            # PyMuPDFLoader plus fiable que PyPDFLoader
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

def load_documents(data_path: str) -> List[Document]:
    if HASH_STORE.exists():
        with open(HASH_STORE, "r") as f:
            ingested_hashes = set(json.load(f))
    else:
        ingested_hashes = set()

    documents = []
    files = list(Path(data_path).rglob("*"))
    logger.info(f"📂 Dossier scanné (récursif) : {data_path}")

    for file in files:
        if not file.is_file():
            continue
        ext = file.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        file_hash = compute_file_hash(file)
        if file_hash in ingested_hashes:
            logger.info(f"⏭️ Déjà traité (hash) : {file.name}")
            continue

        try:
            loader_class = SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(str(file))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file.name
                doc.metadata["file_type"] = ext
                doc.metadata["file_hash"] = file_hash

            documents.extend(docs)
            ingested_hashes.add(file_hash)
            logger.info(f"✅ Chargé : {file.name} ({len(docs)} pages/sections)")

        except OSError as e:
            logger.error(f"❌ Erreur d'accès fichier {file.name} : {e}")
        except (RuntimeError, ValueError, UnicodeDecodeError) as e:
            logger.error(f"❌ Erreur de parsing {file.name} : {e}", exc_info=True)

    with open(HASH_STORE, "w") as f:
        json.dump(list(ingested_hashes), f)

    logger.info(f"📊 Total documents chargés : {len(documents)}")
    return documents