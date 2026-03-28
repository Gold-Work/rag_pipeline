from unittest.mock import patch
from src.query.cache import get_cached, get_index_version, invalidate_cache, make_key, set_cached


# ---------------------------------------------------------------------------
# make_key — normalisation et versioning
# ---------------------------------------------------------------------------

def test_make_key_normalizes_case_and_whitespace():
    with patch("src.query.cache.get_index_version", return_value="abc123"):
        k1 = make_key("  Bonjour Monde  ", 5, [])
        k2 = make_key("bonjour monde", 5, [])
    assert k1 == k2


def test_make_key_includes_top_k():
    with patch("src.query.cache.get_index_version", return_value="abc123"):
        k1 = make_key("question", 3, [])
        k2 = make_key("question", 7, [])
    assert k1 != k2


def test_make_key_includes_index_version():
    with patch("src.query.cache.get_index_version", return_value="v1"):
        k1 = make_key("question", 5, [])
    with patch("src.query.cache.get_index_version", return_value="v2"):
        k2 = make_key("question", 5, [])
    assert k1 != k2


def test_make_key_includes_chunk_ids():
    with patch("src.query.cache.get_index_version", return_value="abc123"):
        k1 = make_key("question", 5, ["chunk_a", "chunk_b"])
        k2 = make_key("question", 5, ["chunk_a", "chunk_c"])
    assert k1 != k2


def test_make_key_chunk_ids_order_insensitive():
    with patch("src.query.cache.get_index_version", return_value="abc123"):
        k1 = make_key("question", 5, ["chunk_b", "chunk_a"])
        k2 = make_key("question", 5, ["chunk_a", "chunk_b"])
    assert k1 == k2


# ---------------------------------------------------------------------------
# get_cached / set_cached — hit et miss
# ---------------------------------------------------------------------------

def test_cache_miss_returns_none():
    with patch("src.query.cache.get_index_version", return_value="abc123"):
        key = make_key("question inconnue xyz 999", 5, [])
    assert get_cached(key) is None


def test_cache_stores_and_retrieves():
    with patch("src.query.cache.get_index_version", return_value="abc123"):
        key = make_key("quelle est la capitale", 5, ["chunk_1"])
    set_cached(key, "Paris")
    assert get_cached(key) == "Paris"


# ---------------------------------------------------------------------------
# get_index_version — lazy compute
# ---------------------------------------------------------------------------

def test_index_version_computed_lazily():
    import src.query.cache as cache_mod

    cache_mod._index_version = None
    with patch("src.query.cache._compute_version", return_value="lazy_v1") as mock_compute:
        v = cache_mod.get_index_version()
    assert v == "lazy_v1"
    assert mock_compute.call_count == 1


def test_index_version_not_recomputed_when_cached():
    import src.query.cache as cache_mod

    cache_mod._index_version = "already_set"
    with patch("src.query.cache._compute_version") as mock_compute:
        v = cache_mod.get_index_version()
    assert v == "already_set"
    mock_compute.assert_not_called()
    cache_mod._index_version = None  # restore


# ---------------------------------------------------------------------------
# invalidate_cache — remet la version à None et vide le cache
# ---------------------------------------------------------------------------

def test_invalidate_cache_resets_version():
    import src.query.cache as cache_mod

    cache_mod._index_version = "stale"
    invalidate_cache()
    assert cache_mod._index_version is None


def test_invalidate_cache_clears_entries():
    import src.query.cache as cache_mod

    with patch("src.query.cache.get_index_version", return_value="abc123"):
        key = make_key("cached question", 5, ["chunk_x"])
    set_cached(key, "stored value")
    assert get_cached(key) == "stored value"

    invalidate_cache()
    assert get_cached(key) is None
