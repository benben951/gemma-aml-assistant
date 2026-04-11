"""Data models"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GemmaResponse:
    content: str
    raw_response: str
    thinking: Optional[str] = None
    sources: List[Dict] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class Document:
    id: str
    content: str
    source: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    document: Document
    score: float
    excerpt: str


__all__ = ['GemmaResponse', 'Document', 'SearchResult']