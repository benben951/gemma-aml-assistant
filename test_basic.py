"""基本功能测试"""
import sys
sys.path.insert(0, 'src')

from rag.retriever import AMLRetriever, Document

def test_retriever():
    """测试检索器"""
    print('创建 AMLRetriever...')
    retriever = AMLRetriever(storage_type='memory')
    print('OK')

    # 添加示例文档
    print('添加文档...')
    docs = [
        Document(content='金融机构应对客户进行尽职调查，了解客户身份和业务性质。', source='AML法规第12条'),
        Document(content='高风险客户需进行增强型尽职调查，包括额外身份验证。', source='KYC指南第3章'),
    ]
    count = retriever.add_documents(docs)
    print(f'添加 {count} 个文档')

    # 搜索测试
    print('搜索测试...')
    results = retriever.search('什么是尽职调查', top_k=2)
    
    for r in results:
        print(f'  来源: {r["source"]}')
        print(f'  内容: {r["content"][:50]}...')
        print(f'  相关度: {r["score"]:.2f}')
        print()

    # Collection 信息
    info = retriever.get_collection_info()
    print(f'Collection: {info["name"]}, 文档数: {info["points_count"]}')
    
    print('测试完成!')
    return True

if __name__ == '__main__':
    test_retriever()