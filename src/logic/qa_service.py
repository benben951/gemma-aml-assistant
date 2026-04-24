"""QA service - orchestrate RAG + LLM + explainability"""

from typing import List, Dict, Any, Optional
from .gemma_client import GemmaClient
from .rag_pipeline import RAGPipeline
from .explainability import ExplainabilityEngine
from ..data.models import GemmaResponse, Document, SearchResult, ExplainedResponse
from ..data.config import settings


class QAService:
    """Orchestrates the full QA pipeline: retrieve → generate → explain."""

    def __init__(
        self,
        gemma_client: GemmaClient,
        rag_pipeline: RAGPipeline,
        explainability: ExplainabilityEngine,
    ):
        self.gemma_client = gemma_client
        self.rag_pipeline = rag_pipeline
        self.explainability = explainability
    
    def answer(self, query: str, enable_thinking: bool = False) -> ExplainedResponse:
        """Answer a query using RAG + LLM with full explainability.

        Returns an ExplainedResponse with answer, thinking, sources,
        confidence, and reasoning_chain.
        """
        # Step 1: Retrieve relevant documents
        search_results = self.rag_pipeline.retrieve(query, top_k=settings.TOP_K_RESULTS)

        # Step 2: Use explainability engine for full analysis
        explained = self.explainability.analyze_with_sources(
            query=query,
            search_results=search_results,
            enable_thinking=enable_thinking,
        )

        return explained
    
    def answer_simple(self, query: str, enable_thinking: bool = False) -> GemmaResponse:
        """Simple answer without explainability metadata. For backward compat."""
        search_results = self.rag_pipeline.retrieve(query, top_k=settings.TOP_K_RESULTS)
        
        context = self._build_context(search_results)
        raw_response = self.gemma_client.generate(
            prompt=query,
            context=context,
            enable_thinking=enable_thinking,
        )
        
        sources = self.explainability.extract_sources(search_results)
        
        return GemmaResponse(
            content=raw_response.content,
            raw_response=raw_response.raw_response,
            thinking=raw_response.thinking,
            sources=sources,
        )
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        context_parts = []
        for result in search_results:
            context_parts.append(
                f"Source: {result.document.source}\nContent: {result.excerpt}"
            )
        return "\n\n".join(context_parts)


__all__ = ['QAService']