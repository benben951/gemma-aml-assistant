"""
RAG 检索模块 - 基于 Qdrant 的向量检索

支持三种存储模式：
- memory: 内存模式（适合测试）
- qdrant_local: 本地文件存储
- qdrant_cloud: Qdrant Cloud 云端存储

设计要点：
- 软依赖：sentence-transformers 可选，不影响模块导入
- 接口兼容 LangChain：search(), get_relevant_documents(), invoke()
"""

import os
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid


class StorageType(Enum):
    MEMORY = "memory"
    QDRANT_LOCAL = "qdrant_local"
    QDRANT_CLOUD = "qdrant_cloud"


@dataclass
class Document:
    """
    文档数据结构
    
    Attributes:
        content: 文档内容
        source: 来源（文件名/URL等）
        metadata: 元数据（章节、页码等）
        doc_id: 文档ID（可选，自动生成）
    """
    content: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())


class EmbeddingAdapter:
    """
    Embedding 适配层
    
    默认使用 sentence-transformers，支持自定义 embedding 函数
    软依赖设计：未安装 sentence-transformers 时可传入自定义 embedding
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 快速、轻量
    DEFAULT_DIMENSION = 384
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_func: Optional[Callable] = None,
        dimension: Optional[int] = None,
    ):
        """
        初始化 Embedding
        
        Args:
            model_name: sentence-transformers 模型名称
            embedding_func: 自定义 embedding 函数（callable）
            dimension: 向量维度（使用自定义函数时必须指定）
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._embedding_func = embedding_func
        self._dimension = dimension
        self._model = None
        
        # 如果没有自定义函数，尝试加载 sentence-transformers
        if embedding_func is None:
            self._load_sentence_transformers()
    
    def _load_sentence_transformers(self):
        """加载 sentence-transformers（软依赖）"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"Embedding 模型加载成功: {self.model_name}, 维度: {self._dimension}")
        except ImportError:
            print(
                "警告: sentence-transformers 未安装。\n"
                "请安装: pip install sentence-transformers\n"
                "或传入自定义 embedding_func 参数。"
            )
            self._dimension = self.DEFAULT_DIMENSION
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本转换为向量
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表
        """
        if self._embedding_func is not None:
            # 使用自定义函数
            return [self._embedding_func(text) for text in texts]
        
        if self._model is not None:
            # 使用 sentence-transformers
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        
        raise RuntimeError(
            "Embedding 未配置。请安装 sentence-transformers 或传入 embedding_func。"
        )
    
    def embed_single(self, text: str) -> List[float]:
        """单个文本的 embedding"""
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension or self.DEFAULT_DIMENSION


class AMLRetriever:
    """
    AML 检索器 - 基于 Qdrant
    
    核心功能：
    - add_documents(): 添加文档到向量库
    - search(): 检索相关文档
    - get_context(): 构建 RAG 上下文
    - invoke(): LangChain 兼容接口
    
    支持三种存储模式：
    - memory: 内存模式（测试）
    - qdrant_local: 本地文件
    - qdrant_cloud: 云端存储
    """
    
    DEFAULT_COLLECTION = "aml_documents"
    
    def __init__(
        self,
        storage_type: str = "memory",
        collection_name: Optional[str] = None,
        embedding: Optional[EmbeddingAdapter] = None,
        persist_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        初始化检索器
        
        Args:
            storage_type: 存储类型 (memory/qdrant_local/qdrant_cloud)
            collection_name: 集合名称
            embedding: Embedding 适配器（可选，默认使用 sentence-transformers）
            persist_path: 本地存储路径（qdrant_local 模式）
            qdrant_url: Qdrant Cloud URL（qdrant_cloud 模式）
            qdrant_api_key: Qdrant Cloud API Key
        """
        self.storage_type = StorageType(storage_type.lower())
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        
        # Embedding
        self.embedding = embedding or EmbeddingAdapter()
        
        # 初始化 Qdrant 客户端
        self._client = None
        self._init_qdrant_client(
            persist_path=persist_path,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
        
        # 确保 collection 存在
        self._ensure_collection()
    
    def _init_qdrant_client(
        self,
        persist_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        """初始化 Qdrant 客户端"""
        try:
            from qdrant_client import QdrantClient
            
            if self.storage_type == StorageType.MEMORY:
                self._client = QdrantClient(location=":memory:")
                print("Qdrant 内存模式初始化成功")
            
            elif self.storage_type == StorageType.QDRANT_LOCAL:
                path = persist_path or "./qdrant_data"
                self._client = QdrantClient(path=path)
                print(f"Qdrant 本地模式初始化成功: {path}")
            
            elif self.storage_type == StorageType.QDRANT_CLOUD:
                if qdrant_url is None:
                    raise ValueError("qdrant_cloud 模式需要提供 qdrant_url")
                self._client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                )
                print(f"Qdrant Cloud 模式初始化成功: {qdrant_url}")
            
        except ImportError as e:
            raise ImportError(
                "qdrant-client 未安装。请安装: pip install qdrant-client"
            ) from e
    
    def _ensure_collection(self):
        """确保 collection 存在"""
        from qdrant_client.http import models
        
        # 检查 collection 是否存在
        if not self._client.collection_exists(self.collection_name):
            # 创建 collection
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            print(
                f"Collection '{self.collection_name}' 创建成功, "
                f"向量维度: {self.embedding.dimension}"
            )
        else:
            # 检查维度是否匹配
            info = self._client.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            
            if existing_dim != self.embedding.dimension:
                raise ValueError(
                    f"Collection '{self.collection_name}' 向量维度 ({existing_dim}) "
                    f"与当前 Embedding 维度 ({self.embedding.dimension}) 不匹配！"
                )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        添加文档到向量库
        
        Args:
            documents: 文档列表
            batch_size: 批量处理大小
        
        Returns:
            成功添加的文档数量
        """
        from qdrant_client.http import models
        
        added_count = 0
        
        # 批量处理
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 生成 embedding
            texts = [doc.content for doc in batch]
            vectors = self.embedding.embed(texts)
            
            # 构建 PointStruct
            points = []
            for doc, vector in zip(batch, vectors):
                points.append(
                    models.PointStruct(
                        id=doc.doc_id,
                        vector=vector,
                        payload={
                            "content": doc.content,
                            "source": doc.source,
                            "metadata": doc.metadata,
                        },
                    )
                )
            
            # 写入 Qdrant
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            added_count += len(batch)
        
        print(f"成功添加 {added_count} 个文档到 '{self.collection_name}'")
        return added_count
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            score_threshold: 最低相似度阈值
        
        Returns:
            文档列表，每个包含 content, source, metadata, score
        """
        # 生成查询向量
        query_vector = self.embedding.embed_single(query)
        
        # 查询 Qdrant
        result = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        
        # 格式化结果
        documents = []
        for point in result.points:
            documents.append({
                "content": point.payload.get("content", ""),
                "source": point.payload.get("source", "unknown"),
                "metadata": point.payload.get("metadata", {}),
                "score": point.score,
                "doc_id": point.id,
            })
        
        return documents
    
    def get_context(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 2000,
    ) -> str:
        """
        构建 RAG 上下文
        
        Args:
            query: 查询文本
            top_k: 检索文档数量
            max_tokens: 最大上下文长度（估算）
        
        Returns:
            格式化的上下文文本
        """
        documents = self.search(query, top_k=top_k)
        
        if not documents:
            return "未找到相关文档。"
        
        # 构建上下文
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # 添加来源标注
            source = doc["source"]
            content = doc["content"]
            score = doc.get("score", 0.0)
            
            part = f"\n[{i+1}] {source} (相关度: {score:.2f})\n{content}\n"
            
            # 检查长度限制
            estimated_tokens = len(part) // 4  # 简单估算
            
            if current_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(part)
            current_length += estimated_tokens
        
        return "".join(context_parts)
    
    def invoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        LangChain 兼容接口
        
        Args:
            input: 查询文本或字典 {"query": str, "top_k": int}
            config: 配置参数
        
        Returns:
            文档列表
        """
        # 解析输入
        if isinstance(input, str):
            query = input
            top_k = 5
        elif isinstance(input, dict):
            query = input.get("query", "")
            top_k = input.get("top_k", 5)
        else:
            raise ValueError(f"不支持的输入类型: {type(input)}")
        
        return self.search(query, top_k=top_k)
    
    def get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        """
        LangChain 兼容接口（返回 Document 对象）
        
        Args:
            query: 查询文本
        
        Returns:
            Document 对象列表
        """
        results = self.search(query)
        
        documents = []
        for result in results:
            documents.append(
                Document(
                    content=result["content"],
                    source=result["source"],
                    metadata=result.get("metadata", {}),
                    doc_id=result.get("doc_id"),
                )
            )
        
        return documents
    
    def delete_collection(self) -> bool:
        """删除 collection"""
        return self._client.delete_collection(self.collection_name)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取 collection 信息"""
        info = self._client.get_collection(self.collection_name)
        
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "status": info.status.value,
        }


# RAG Pipeline（完整流程）
class RAGPipeline:
    """
    RAG 完整流程
    
    检索 + 生成 + 解释
    
    使用示例:
        pipeline = RAGPipeline(gemma_client, retriever)
        result = pipeline.query("什么是AML尽职调查？")
        print(result["answer"])
        print(result["sources"])
    """
    
    def __init__(
        self,
        gemma_client,  # GemmaClient 实例
        retriever: AMLRetriever,
        explainability_engine=None,  # ExplainabilityEngine 实例（可选）
        top_k: int = 5,
        enable_thinking: bool = True,
    ):
        """
        初始化 RAG Pipeline
        
        Args:
            gemma_client: Gemma 4 客户端
            retriever: 检索器
            explainability_engine: 可解释性引擎（可选）
            top_k: 检索文档数量
            enable_thinking: 是否启用 Thinking 模式
        """
        self.client = gemma_client
        self.retriever = retriever
        self.explainability = explainability_engine
        self.top_k = top_k
        self.enable_thinking = enable_thinking
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        执行 RAG 查询
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
        
        Returns:
            包含 answer, thinking, sources 的结果
        """
        k = top_k or self.top_k
        
        # 1. 检索
        documents = self.retriever.search(question, top_k=k)
        
        # 2. 构建上下文
        context = self.retriever.get_context(question, top_k=k)
        
        # 3. 生成回答
        response = self.client.generate(
            prompt=question,
            context=context,
            enable_thinking=self.enable_thinking,
        )
        
        # 4. 如果有可解释性引擎，使用它格式化
        if self.explainability:
            from src.safety.explainability import ExplainedResponse
            
            explained = self.explainability.analyze_with_sources(
                query=question,
                retrieved_docs=documents,
                enable_thinking=self.enable_thinking,
            )
            
            return {
                "answer": explained.answer,
                "thinking": explained.thinking,
                "sources": explained.sources,
                "confidence": explained.confidence,
                "reasoning_chain": explained.reasoning_chain,
            }
        
        # 5. 简单返回
        return {
            "answer": response.content,
            "thinking": response.thinking,
            "sources": documents,
            "confidence": 0.7 if documents else 0.3,  # 简单估算
        }


# 便捷函数
def create_retriever(
    storage_type: str = "memory",
    collection_name: str = "aml_documents",
    **kwargs
) -> AMLRetriever:
    """创建检索器的便捷函数"""
    return AMLRetriever(
        storage_type=storage_type,
        collection_name=collection_name,
        **kwargs
    )


def create_rag_pipeline(
    gemma_client,
    storage_type: str = "memory",
    enable_thinking: bool = True,
    **kwargs
) -> RAGPipeline:
    """创建 RAG Pipeline 的便捷函数"""
    retriever = create_retriever(storage_type=storage_type, **kwargs)
    
    return RAGPipeline(
        gemma_client=gemma_client,
        retriever=retriever,
        enable_thinking=enable_thinking,
    )


if __name__ == "__main__":
    # 测试示例
    print("AML Retriever 测试")
    
    # 创建检索器（内存模式）
    retriever = AMLRetriever(storage_type="memory")
    
    # 添加示例文档
    docs = [
        Document(
            content="金融机构应对客户进行尽职调查，了解客户身份、业务性质和交易目的。",
            source="AML法规第12条",
        ),
        Document(
            content="高风险客户需进行增强型尽职调查，包括额外的身份验证和交易监控。",
            source="KYC指南第3章",
        ),
    ]
    
    retriever.add_documents(docs)
    
    # 搜索
    results = retriever.search("什么是尽职调查？", top_k=2)
    
    for doc in results:
        print(f"\n来源: {doc['source']}")
        print(f"内容: {doc['content'][:100]}...")
        print(f"相关度: {doc['score']:.2f}")