"""快速测试 - verify core data structures"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.models import GemmaResponse, Document, SearchResult


def test_models():
    # 测试 GemmaResponse
    r = GemmaResponse(content="测试回答", raw_response="原始响应", thinking="推理过程")
    assert r.content == "测试回答"
    assert r.thinking == "推理过程"
    assert r.sources == []
    print("GemmaResponse OK!")

    # 测试 Document
    d = Document(content="测试文档", source="test.txt")
    assert d.doc_id is not None  # auto-generated
    assert d.content == "测试文档"
    print("Document OK!")

    # 测试 SearchResult
    sr = SearchResult(document=d, score=0.95, excerpt="测试摘要")
    assert sr.score == 0.95
    print("SearchResult OK!")

    print("\n基本数据结构测试通过!")


if __name__ == "__main__":
    test_models()