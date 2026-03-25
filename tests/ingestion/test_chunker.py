from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents

def test_chunker_creates_chunks():
    docs = [Document(page_content="word " * 300, metadata={"source_file": "test.pdf"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1

def test_chunker_chunk_ids_unique():
    docs = [Document(page_content="word " * 300, metadata={"source_file": "test.pdf"})]
    chunks = chunk_documents(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))

def test_chunker_metadata_preserved():
    docs = [Document(page_content="word " * 300, metadata={"source_file": "test.pdf"})]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "chunk_size" in chunk.metadata
        assert chunk.metadata["source_file"] == "test.pdf"

def test_chunker_empty_input():
    result = chunk_documents([])
    assert result == []