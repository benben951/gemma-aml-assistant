"""快速测试 - verify core data structures"""

import sys
sys.path.insert(0, "src")

from src.data.models import GemmaResponse, Document, SearchResult

# 测试 GemmaResponse
r = GemmaResponse(content="测试回答", raw_response="原始响应", thinking="推理过程")
print(f"content: {r.content}")
print(f"thinking: {r.thinking}")
print(f"sources: {r.sources}")
print("GemmaResponse OK!")

# 测试 Document
d = Document(content="测试文档", source="test.txt")
print(f"doc_id: {d.doc_id}")
print(f"content: {d.content}")
print("Document OK!")

# 测试 SearchResult
sr = SearchResult(document=d, score=0.95, excerpt="测试摘要")
print(f"score: {sr.score}")
print(f"excerpt: {sr.excerpt}")
print("SearchResult OK!")

print("\n基本数据结构测试通过!")