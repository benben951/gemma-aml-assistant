"""基本功能测试"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.logic.rag_pipeline import RAGPipeline
from src.data.models import Document


def test_retriever():
    """测试检索器"""
    print("创建 RAGPipeline...")
    pipeline = RAGPipeline(storage_type="memory")
    print("OK")

    # 添加示例文档
    print("添加文档...")
    docs = [
        Document(
            content="金融机构应对客户进行尽职调查，了解客户身份和业务性质。",
            source="AML法规第12条",
        ),
        Document(
            content="高风险客户需进行增强型尽职调查，包括额外身份验证。",
            source="KYC指南第3章",
        ),
    ]
    count = pipeline.add_documents(docs)
    print(f"添加 {count} 个文档chunks")

    # 搜索测试
    print("搜索测试...")
    results = pipeline.retrieve("什么是尽职调查", top_k=2)

    for r in results:
        print(f"  来源: {r.document.source}")
        print(f"  内容: {r.excerpt[:50]}...")
        print(f"  相关度: {r.score:.2f}")
        print()

    # Collection 信息
    info = pipeline.get_collection_info()
    print(f"Collection info: {info}")

    print("测试完成!")
    return True


if __name__ == "__main__":
    test_retriever()