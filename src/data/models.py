"""Data models - unified models for the entire application"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import uuid


@dataclass
class GemmaResponse:
    """Response from the Gemma LLM client."""
    content: str
    raw_response: str
    thinking: Optional[str] = None
    sources: List[Dict] = field(default_factory=list)


@dataclass
class Document:
    """A document with content and metadata."""
    content: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())


@dataclass
class SearchResult:
    """A search result with relevance score."""
    document: Document
    score: float
    excerpt: str


@dataclass
class ExplainedResponse:
    """An answer with explainability metadata."""
    answer: str
    thinking: Optional[str] = None
    sources: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)


__all__ = ["GemmaResponse", "Document", "SearchResult", "ExplainedResponse"]