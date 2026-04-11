"""Vector store management"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum


class StorageType(Enum):
    MEMORY = "memory"
    QDRANT_LOCAL = "qdrant_local"
    QDRANT_CLOUD = "qdrant_cloud"


class VectorStore:
    def __init__(
        self,
        storage_type: StorageType = StorageType.MEMORY,
        persist_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "aml_knowledge",
        vector_size: int = 768,
    ):
        self.storage_type = storage_type
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None
        self._init_client(persist_path, qdrant_url, qdrant_api_key)
    
    def _init_client(self, persist_path, qdrant_url, qdrant_api_key):
        try:
            from qdrant_client import QdrantClient
            
            if self.storage_type == StorageType.MEMORY:
                self._client = QdrantClient(location=":memory:")
            elif self.storage_type == StorageType.QDRANT_LOCAL:
                path = persist_path or "./qdrant_data"
                self._client = QdrantClient(path=path)
            elif self.storage_type == StorageType.QDRANT_CLOUD:
                if qdrant_url is None:
                    raise ValueError("qdrant_cloud requires qdrant_url")
                self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            
            self._ensure_collection()
            
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")
    
    def _ensure_collection(self):
        from qdrant_client.http import models
        
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
    
    def insert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        from qdrant_client.http import models
        
        points = [
            models.PointStruct(
                id=i,
                vector=vectors[i],
                payload=metadata[i],
            )
            for i in range(len(vectors))
        ]
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]
    
    def delete_collection(self):
        self._client.delete_collection(self.collection_name)


__all__ = ['VectorStore', 'StorageType']