"""Explainability engine - provides confidence scoring and source citation"""

from typing import Dict, Any, List, Optional

from ..data.models import ExplainedResponse, SearchResult


class ExplainabilityEngine:
    def __init__(self, gemma_client):
        self.client = gemma_client
    
    def analyze_with_sources(
        self,
        query: str,
        search_results: List[SearchResult],
        enable_thinking: bool = True,
    ) -> ExplainedResponse:
        """Analyze a query using retrieved sources with explainability."""
        context = self._build_context(search_results)
        
        result = self.client.analyze_with_thinking(
            query=query,
            context=context,
            enable_thinking=enable_thinking,
        )
        
        confidence = self._evaluate_confidence(result, search_results)
        reasoning_chain = self._build_reasoning_chain(result.get('thinking', ''))
        
        return ExplainedResponse(
            answer=result['answer'],
            thinking=result['thinking'],
            sources=self._format_sources(search_results),
            confidence=confidence,
            reasoning_chain=reasoning_chain,
        )
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        context_parts = []
        
        for i, sr in enumerate(search_results[:5]):
            context_parts.append(
                f"[{i+1}] {sr.document.source}\n{sr.document.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_sources(self, search_results: List[SearchResult]) -> List[Dict]:
        sources = []
        
        for sr in search_results[:3]:
            sources.append({
                "source": sr.document.source,
                "excerpt": sr.excerpt[:200] if sr.excerpt else "",
                "score": sr.score,
                "page": sr.document.metadata.get("page"),
            })
        
        return sources
    
    def _evaluate_confidence(
        self, result: Dict, search_results: List[SearchResult]
    ) -> float:
        if not search_results:
            return 0.3
        
        scores = [sr.score for sr in search_results[:3]]
        avg_score = sum(scores) / len(scores)
        
        answer = result.get('answer', '')
        
        cited_sources = 0
        for sr in search_results[:3]:
            source_name = sr.document.source.lower()
            if source_name and source_name in answer.lower():
                cited_sources += 1
        
        citation_factor = cited_sources / len(search_results[:3]) if search_results[:3] else 0
        
        confidence = 0.6 * avg_score + 0.4 * citation_factor
        
        return min(max(confidence, 0.0), 1.0)
    
    def _build_reasoning_chain(self, thinking: str) -> List[str]:
        if not thinking:
            return []
        
        steps = []
        for line in thinking.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                steps.append(line)
        
        return steps[:5]


__all__ = ['ExplainabilityEngine']