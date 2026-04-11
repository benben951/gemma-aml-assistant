"""RAG pipeline with Qdrant"""

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
    content: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())


class EmbeddingAdapter:
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_func: Optional[Callable] = None,
        dimension: Optional[int] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._embedding_func = embedding_func
        self._dimension = dimension
        self._model = None
        
        if embedding_func is None:
            self._load_sentence_transformers()
    
    def _load_sentence_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except ImportError:
            self._dimension = self.DEFAULT_DIMENSION
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._embedding_func:
            return self._embedding_func(texts)
        
        if self._model:
            embeddings = self._model.encode(texts)
            return embeddings.tolist()
        
        import random
        return [[random.random() for _ in range(self._dimension)] for _ in texts]
    
    def dimension(self) -> int:
        return self._dimension


class RAGPipeline:
    def __init__(
        self,
        storage_type: str = "memory",
        persist_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "aml_knowledge",
        embedding_model: Optional[str] = None,
        embedding_func: Optional[Callable] = None,
        top_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.storage_type = StorageType(storage_type.lower())
        self.collection_name = collection_name
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embedding = EmbeddingAdapter(
            model_name=embedding_model,
            embedding_func=embedding_func,
        )
        
        self._init_qdrant(persist_path, qdrant_url, qdrant_api_key)
    
    def _init_qdrant(self, persist_path, qdrant_url, qdrant_api_key):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            if self.storage_type == StorageType.MEMORY:
                self._client = QdrantClient(location=":memory:")
            elif self.storage_type == StorageType.QDRANT_LOCAL:
                path = persist_path or "./qdrant_data"
                self._client = QdrantClient(path=path)
            elif self.storage_type == StorageType.QDRANT_CLOUD:
                if qdrant_url is None:
                    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            
            self._ensure_collection()
            
        except ImportError:
            self._client = None
    
    def _ensure_collection(self):
        from qdrant_client.http import models
        
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding.dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
    
    def add_documents(self, documents: List[Document]) -> int:
        if not self._client:
            return 0
        
        chunks = []
        chunk_metadata = []
        
        for doc in documents:
            doc_chunks = self._chunk_text(doc.content, doc.source, doc.metadata)
            chunks.extend(doc_chunks)
        
        if not chunks:
            return 0
        
        vectors = self.embedding.embed([c.content for c in chunks])
        
        from qdrant_client.http import models
        
        points = [
            models.PointStruct(
                id=c.doc_id,
                vector=vectors[i],
                payload={
                    "content": c.content,
                    "source": c.source,
                    "metadata": c.metadata,
                },
            )
            for i, c in enumerate(chunks)
        ]
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        return len(chunks)
    
    def _chunk_text(self, text: str, source: str, metadata: Dict) -> List[Document]:
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_content = text[i:i + self.chunk_size]
            
            if chunk_content.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i // (self.chunk_size - self.chunk_overlap)
                
                chunks.append(Document(
                    content=chunk_content,
                    source=source,
                    metadata=chunk_metadata,
                ))
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = None) -> List:
        if not self._client:
            return []
        
        k = top_k or self.top_k
        
        query_vector = self.embedding.embed([query])[0]
        
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
        )
        
        from ..data.models import SearchResult
        
        search_results = []
        for r in results:
            payload = r.payload
            doc = Document(
                doc_id=str(r.id),
                content=payload.get("content", ""),
                source=payload.get("source", "unknown"),
                metadata=payload.get("metadata", {}),
            )
            search_results.append(SearchResult(
                document=doc,
                score=r.score,
                excerpt=payload.get("content", "")[:200],
            ))
        
        return search_results
    
    def get_collection_info(self) -> Dict:
        if not self._client:
            return {"status": "not_initialized"}
        
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "status": "ready",
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }
        except Exception:
            return {"status": "empty"}
    
    def delete_collection(self):
        if self._client:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass


__all__ = ['RAGPipeline', 'Document', 'StorageType']