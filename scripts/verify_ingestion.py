import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import chromadb
from collections import defaultdict
from src.utils.logger import get_logger

logger = get_logger("verify_ingestion")


def verify():
    persist_path = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    client = chromadb.PersistentClient(path=persist_path)
    col = client.get_collection(collection_name)

    logger.info("=" * 60)
    logger.info("🔍 VÉRIFICATION DU VECTOR STORE")
    logger.info("=" * 60)

    # Total
    total = col.count()
    logger.info(f"📊 Total chunks indexés : {total}")

    # Récupérer tout
    results = col.get(include=["documents", "metadatas"])
    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    # Stats par fichier
    logger.info("\n📁 CHUNKS PAR FICHIER :")
    stats = defaultdict(int)
    for meta in metas:
        source = meta.get("source_file", "inconnu")
        stats[source] += 1
    for source, count in sorted(stats.items()):
        logger.info(f"   📄 {source} → {count} chunks")

    # Taille des chunks
    sizes = [len(doc) for doc in docs]
    logger.info("\n📏 TAILLE DES CHUNKS :")
    logger.info(f"   Min  : {min(sizes)} chars")
    logger.info(f"   Max  : {max(sizes)} chars")
    logger.info(f"   Moy  : {int(sum(sizes)/len(sizes))} chars")

    # Vérification IDs dupliqués
    logger.info("\n🔑 VÉRIFICATION IDs :")
    unique_ids = set(ids)
    if len(unique_ids) == len(ids):
        logger.info(f"   ✅ Tous les IDs sont uniques ({len(ids)})")
    else:
        logger.warning(f"   ⚠️  {len(ids) - len(unique_ids)} IDs dupliqués détectés !")

    # Vérification chunks vides
    empty = [i for i, doc in enumerate(docs) if not doc or len(doc.strip()) < 10]
    if empty:
        logger.warning(f"   ⚠️  {len(empty)} chunks vides ou trop courts détectés")
    else:
        logger.info("   ✅ Aucun chunk vide")

    # Aperçu de 3 chunks aléatoires
    logger.info("\n👀 APERÇU DE 3 CHUNKS :")
    import random

    samples = random.sample(range(len(ids)), min(3, len(ids)))
    for i in samples:
        logger.info(f"\n   --- Chunk {i} ---")
        logger.info(f"   ID       : {ids[i]}")
        logger.info(f"   Source   : {metas[i].get('source_file', '?')}")
        logger.info(f"   Taille   : {len(docs[i])} chars")
        logger.info(f"   Aperçu   : {docs[i][:150].strip()}...")

    logger.info("\n" + "=" * 60)
    logger.info("✅ VÉRIFICATION TERMINÉE")
    logger.info("=" * 60)


if __name__ == "__main__":
    verify()
