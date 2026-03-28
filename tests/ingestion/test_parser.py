from langchain_core.documents import Document
from src.ingestion.parser import parse_documents


def test_parser_removes_short_pages():
    docs = [
        Document(page_content="ok", metadata={"source_file": "test.pdf"}),
        Document(page_content="x" * 100, metadata={"source_file": "test.pdf"}),
    ]
    result = parse_documents(docs)
    assert len(result) == 1
    assert len(result[0].page_content) == 100


def test_parser_cleans_whitespace():
    content = "hello   world\n\n\nfoo " * 20
    docs = [Document(page_content=content, metadata={"source_file": "test.pdf"})]
    result = parse_documents(docs)
    assert len(result) == 1
    assert "   " not in result[0].page_content


def test_parser_empty_input():
    result = parse_documents([])
    assert result == []


def test_parser_preserves_metadata():
    docs = [Document(page_content="x" * 100, metadata={"source_file": "test.pdf", "page": 1})]
    result = parse_documents(docs)
    assert result[0].metadata["source_file"] == "test.pdf"
