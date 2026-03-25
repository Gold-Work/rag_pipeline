from unittest.mock import patch, MagicMock
from src.query.rewriter import rewrite_query

def test_rewriter_returns_original_on_failure():
    with patch("src.query.rewriter.client") as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = rewrite_query("test question")
        assert result == ["test question"]

def test_rewriter_deduplicates():
    with patch("src.query.rewriter.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "test question\nautre question"
        mock_client.chat.completions.create.return_value = mock_response
        result = rewrite_query("test question")
        assert result.count("test question") == 1

def test_rewriter_original_always_first():
    with patch("src.query.rewriter.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "variante 1\nvariante 2"
        mock_client.chat.completions.create.return_value = mock_response
        result = rewrite_query("ma question")
        assert result[0] == "ma question"