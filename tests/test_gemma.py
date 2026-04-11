"""
Gemma Client 单元测试

测试内容：
1. 多 Backend 支持
2. Thinking 模式
3. 响应解析
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGemmaClient:
    """GemmaClient 测试类"""
    
    @pytest.mark.skip(reason="kagglehub 模块仅可在 Kaggle Notebook 环境中使用")
    @patch('kagglehub.model_download')
    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_init_kaggle_backend(self, mock_model, mock_processor, mock_kagglehub):
        """测试 Kaggle backend 初始化"""
        from src.core.gemma_client import GemmaClient
        
        # Mock 返回值
        mock_kagglehub.model_download.return_value = "/mock/path"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock(device=Mock())
        
        client = GemmaClient(backend="kaggle")
        
        assert client.backend.value == "kaggle"
        assert client.model == "gemma-4-26b-a4b"
    
    def test_parse_response_with_thinking(self):
        """测试 Thinking 响应解析"""
        from src.core.gemma_client import GemmaClient
        
        # 创建一个 mock client（不初始化 backend）
        client = GemmaClient.__new__(GemmaClient)
        
        # 测试数据
        raw_response = """<|think|>
这是推理过程第一步。
这是推理过程第二步。
<|/think|>
这是最终回答内容。<|end|>"""
        
        result = client._parse_response(raw_response, enable_thinking=True)
        
        assert result.thinking == "这是推理过程第一步。\n这是推理过程第二步。"
        assert result.content == "这是最终回答内容。"
    
    def test_parse_response_without_thinking(self):
        """测试普通响应解析"""
        from src.core.gemma_client import GemmaClient
        
        client = GemmaClient.__new__(GemmaClient)
        
        raw_response = "这是普通回答内容。<|end|>"
        
        result = client._parse_response(raw_response, enable_thinking=False)
        
        assert result.thinking is None
        assert result.content == "这是普通回答内容。"
    
    def test_default_config(self):
        """测试默认配置"""
        from src.core.gemma_client import GemmaClient
        
        # 验证默认值
        assert GemmaClient.DEFAULT_MODEL == "gemma-4-26b-a4b"
        assert GemmaClient.DEFAULT_TEMPERATURE == 1.0
        assert GemmaClient.DEFAULT_TOP_P == 0.95
        assert GemmaClient.DEFAULT_TOP_K == 64


class TestGemmaResponse:
    """GemmaResponse 测试"""
    
    def test_response_creation(self):
        """测试响应创建"""
        from src.core.gemma_client import GemmaResponse
        
        response = GemmaResponse(
            thinking="推理过程",
            content="最终回答",
            raw_response="原始响应",
        )
        
        assert response.thinking == "推理过程"
        assert response.content == "最终回答"
        assert response.sources == []  # 默认空列表
    
    def test_response_with_sources(self):
        """测试带来源的响应"""
        from src.core.gemma_client import GemmaResponse
        
        sources = [{"source": "AML法规", "excerpt": "内容摘要"}]
        
        response = GemmaResponse(
            content="回答内容",
            raw_response="原始响应内容",
            sources=sources,
        )
        
        assert len(response.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])