"""Configuration"""

import os
from typing import Optional


class Settings:
    DEFAULT_MODEL: str = "gemma-4-26b-a4b"
    KAGGLE_MODEL_PATH: str = "google/gemma-4/transformers/gemma-4-26b-a4b"
    
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "aml_knowledge")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    TOP_K_RESULTS: int = 5
    MIN_SIMILARITY_SCORE: float = 0.7
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 50
    
    MAX_THINKING_TOKENS: int = 1000
    MAX_RESPONSE_TOKENS: int = 500
    
    @classmethod
    def from_env(cls) -> "Settings":
        return cls()


settings = Settings.from_env()


__all__ = ['Settings', 'settings']