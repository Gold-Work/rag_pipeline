import hashlib
import threading
from cachetools import TTLCache
import chromadb
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("cache")
config = get_config()

_cache: TTLCache = TTLCache(
    maxsize=config["cache"]["max_size"],
    ttl=config["cache"]["ttl_seconds"],
)
_cache_lock = threading.Lock()
_version_lock = threading.Lock()

# MD5 hash of sorted chunk IDs — the logical fingerprint of the index.
# None means "not yet computed"; reset to None on explicit invalidation.
_index_version: str | None = None


def _compute_version() -> str:
    """Hash the sorted list of chunk IDs in ChromaDB.

    Returns a short hex digest that changes if and only if the set of indexed
    chunks changes.  Falls back to 'unavailable' if ChromaDB is unreachable.
    """
    try:
        client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
        collection = client.get_or_create_collection(config["embedding"]["collection_name"])
        ids = sorted(collection.get(include=[])["ids"])
        digest = hashlib.md5("|".join(ids).encode()).hexdigest()[:16]
        logger.info(f"🔑 Version index : {digest} ({len(ids)} chunks)")
        return digest
    except Exception as e:
        logger.warning(f"⚠️ Impossible de calculer la version de l'index : {e}")
        return "unavailable"


def get_index_version() -> str:
    """Return the current index version, computing it lazily on first call.

    The version is a pure logical signal (MD5 of sorted chunk IDs).  It is
    reset to None by invalidate_cache() whenever the caller knows the index
    has changed, triggering a recompute on the next request.
    """
    global _index_version
    if _index_version is None:
        with _version_lock:
            if _index_version is None:  # double-checked locking
                _index_version = _compute_version()
    return _index_version


def invalidate_cache() -> None:
    """Reset the cached index version and clear all cache entries.

    Call this after any index mutation (document ingestion, deletion) so the
    next request recomputes the version from ChromaDB and misses the cache.
    """
    global _index_version
    with _version_lock:
        _index_version = None
    with _cache_lock:
        _cache.clear()
    logger.info("🔄 Cache invalidé — index modifié")


def make_key(question: str, top_k: int, chunk_ids: list[str]) -> str:
    """Build a content-addressed cache key.

    Includes:
    - normalised question  (case- and whitespace-insensitive)
    - top_k               (different reranking depth → different result)
    - index version       (MD5 of all chunk IDs — bulk invalidation signal)
    - ids_digest          (MD5 of the retrieved chunk IDs — fine-grained signal:
                           same question but different retrieval → different key)
    """
    version = get_index_version()
    ids_digest = hashlib.md5("|".join(sorted(chunk_ids)).encode()).hexdigest()[:16]
    return f"{question.strip().lower()}|{top_k}|{version}|{ids_digest}"


def get_cached(key: str):
    with _cache_lock:
        entry = _cache.get(key)
    if entry is not None:
        logger.info(f"💾 Cache hit — clé : {key[:80]}")
    return entry


def set_cached(key: str, value) -> None:
    with _cache_lock:
        _cache[key] = value
