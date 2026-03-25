from src.utils.config import get_config

def test_config_loads():
    config = get_config()
    assert config is not None

def test_config_retrieval_keys():
    config = get_config()
    assert "chunk_size" in config["retrieval"]
    assert "chunk_overlap" in config["retrieval"]
    assert "top_k_retrieval" in config["retrieval"]
    assert "top_k_rerank" in config["retrieval"]

def test_config_llm_keys():
    config = get_config()
    assert "model" in config["llm"]
    assert "temperature" in config["llm"]
    assert "max_tokens" in config["llm"]

def test_config_values_valid():
    config = get_config()
    assert config["retrieval"]["chunk_size"] > 0
    assert config["retrieval"]["chunk_overlap"] >= 0
    assert config["retrieval"]["top_k_retrieval"] > 0
    assert 0.0 <= config["llm"]["temperature"] <= 1.0
