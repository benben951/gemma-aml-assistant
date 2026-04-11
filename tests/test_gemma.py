"""Gemma Client tests"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGemmaClient:
    
    @pytest.mark.skip(reason="Kaggle backend only works in Kaggle Notebook")
    @patch('kagglehub.model_download')
    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_init_kaggle_backend(self, mock_model, mock_processor, mock_kagglehub):
        from src.logic.gemma_client import GemmaClient
        
        mock_kagglehub.model_download.return_value = "/mock/path"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock(device=Mock())
        
        client = GemmaClient(backend="kaggle")
        
        assert client.backend.value == "kaggle"
        assert client.model == "gemma-4-26b-a4b"
    
    def test_parse_response_with_thinking(self):
        from src.logic.gemma_client import GemmaClient
        
        client = GemmaClient.__new__(GemmaClient)
        
        raw_response = """<|think|>
Step 1: reasoning.
Step 2: analysis.
<|/think|>
Final answer here.<|end|>"""
        
        result = client._parse_response(raw_response, enable_thinking=True)
        
        assert result.thinking == "Step 1: reasoning.\nStep 2: analysis."
        assert result.content == "Final answer here."
    
    def test_parse_response_without_thinking(self):
        from src.logic.gemma_client import GemmaClient
        
        client = GemmaClient.__new__(GemmaClient)
        raw_response = "Simple answer.<|end|>"
        
        result = client._parse_response(raw_response, enable_thinking=False)
        
        assert result.thinking is None
        assert result.content == "Simple answer."
    
    def test_default_config(self):
        from src.logic.gemma_client import GemmaClient
        
        assert GemmaClient.DEFAULT_MODEL == "gemma-4-26b-a4b"
        assert GemmaClient.DEFAULT_TEMPERATURE == 1.0
        assert GemmaClient.DEFAULT_TOP_P == 0.95
        assert GemmaClient.DEFAULT_TOP_K == 64


class TestGemmaResponse:
    
    def test_response_creation(self):
        from src.data.models import GemmaResponse
        
        response = GemmaResponse(
            thinking="reasoning",
            content="answer",
            raw_response="raw",
        )
        
        assert response.thinking == "reasoning"
        assert response.content == "answer"
        assert response.sources == []
    
    def test_response_with_sources(self):
        from src.data.models import GemmaResponse
        
        sources = [{"source": "doc.pdf", "excerpt": "excerpt text"}]
        
        response = GemmaResponse(
            content="answer",
            raw_response="raw",
            sources=sources,
        )
        
        assert len(response.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])