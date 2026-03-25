from langchain_core.documents import Document
from src.query.augmenter import build_prompt

def test_augmenter_builds_prompt():
    docs = [Document(page_content="contenu test", metadata={})]
    prompt = build_prompt("ma question", docs)
    assert "ma question" in prompt
    assert "contenu test" in prompt

def test_augmenter_empty_docs():
    prompt = build_prompt("ma question", [])
    assert "ma question" in prompt

def test_augmenter_multiple_docs():
    docs = [
        Document(page_content="chunk 1", metadata={}),
        Document(page_content="chunk 2", metadata={}),
    ]
    prompt = build_prompt("question", docs)
    assert "chunk 1" in prompt
    assert "chunk 2" in prompt