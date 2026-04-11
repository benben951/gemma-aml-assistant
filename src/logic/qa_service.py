"""QA service"""

from typing import List, Dict, Any
from .gemma_client import GemmaClient
from .rag_pipeline import RAGPipeline
from .explainability import ExplainabilityEngine
from ..data.models import GemmaResponse, Document, SearchResult
from ..data.config import settings


class QAService:
    def __init__(
        self,
        gemma_client: GemmaClient,
        rag_pipeline: RAGPipeline,
        explainability: ExplainabilityEngine,
    ):
        self.gemma_client = gemma_client
        self.rag_pipeline = rag_pipeline
        self.explainability = explainability
    
    def answer(self, query: str, enable_thinking: bool = False) -> GemmaResponse:
        search_results = self.rag_pipeline.retrieve(query, top_k=settings.TOP_K_RESULTS)
        
        context = self._build_context(search_results)
        raw_response = self.gemma_client.generate(
            query=query,
            context=context,
            enable_thinking=enable_thinking,
        )
        
        sources = self.explainability.extract_sources(search_results)
        confidence = self.explainability.calculate_confidence(search_results)
        
        return GemmaResponse(
            content=raw_response.content,
            raw_response=raw_response.raw_response,
            thinking=raw_response.thinking,
            sources=sources,
        )
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        context_parts = []
        for result in search_results:
            context_parts.append(f"Source: {result.document.source}\nContent: {result.excerpt}")
        return "\n\n".join(context_parts)


__all__ = ['QAService']