"""快速测试"""
import sys
sys.path.insert(0, 'src')

from core.gemma_client import GemmaResponse
from rag.retriever import Document

# 测试 GemmaResponse
r = GemmaResponse(content='测试回答', raw_response='原始响应', thinking='推理过程')
print(f'content: {r.content}')
print(f'thinking: {r.thinking}')
print(f'sources: {r.sources}')
print('GemmaResponse OK!')

# 测试 Document
d = Document(content='测试文档', source='test.txt')
print(f'doc_id: {d.doc_id}')
print(f'content: {d.content}')
print('Document OK!')

print('基本数据结构测试通过!')