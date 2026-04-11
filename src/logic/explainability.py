"""Explainability engine"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExplainedResponse:
    answer: str
    thinking: Optional[str] = None
    sources: List[Dict] = None
    confidence: float = 0.0
    reasoning_chain: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.reasoning_chain is None:
            self.reasoning_chain = []


class ExplainabilityEngine:
    def __init__(self, gemma_client):
        self.client = gemma_client
    
    def analyze_with_sources(
        self,
        query: str,
        retrieved_docs: List[Dict],
        enable_thinking: bool = True,
    ) -> ExplainedResponse:
        context = self._build_context(retrieved_docs)
        
        result = self.client.analyze_with_thinking(
            query=query,
            context=context,
        )
        
        confidence = self._evaluate_confidence(result, retrieved_docs)
        reasoning_chain = self._build_reasoning_chain(result.get('thinking', ''))
        
        return ExplainedResponse(
            answer=result['answer'],
            thinking=result['thinking'],
            sources=self._format_sources(retrieved_docs),
            confidence=confidence,
            reasoning_chain=reasoning_chain,
        )
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:5]):
            source = doc.get('source', 'Unknown')
            content = doc.get('content', '')
            context_parts.append(f"[{i+1}] {source}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        sources = []
        
        for doc in retrieved_docs[:3]:
            sources.append({
                "source": doc.get('source', 'Unknown'),
                "excerpt": doc.get('content', '')[:200],
                "page": doc.get('metadata', {}).get('page'),
            })
        
        return sources
    
    def _evaluate_confidence(self, result: Dict, retrieved_docs: List[Dict]) -> float:
        if not retrieved_docs:
            return 0.3
        
        scores = [doc.get('score', 0.5) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores)
        
        answer = result.get('answer', '')
        
        cited_sources = 0
        for doc in retrieved_docs:
            source_name = doc.get('source', '').lower()
            if source_name and source_name in answer.lower():
                cited_sources += 1
        
        citation_factor = cited_sources / len(retrieved_docs[:3]) if retrieved_docs else 0
        
        confidence = 0.6 * avg_score + 0.4 * citation_factor
        
        return min(max(confidence, 0.0), 1.0)
    
    def _build_reasoning_chain(self, thinking: str) -> List[str]:
        if not thinking:
            return []
        
        steps = []
        lines = thinking.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                steps.append(line)
        
        return steps[:5]
    
    def extract_sources(self, search_results) -> List[Dict]:
        sources = []
        for result in search_results:
            if hasattr(result, 'document'):
                sources.append({
                    "source": result.document.source,
                    "excerpt": result.excerpt[:200] if result.excerpt else "",
                    "confidence": result.score,
                })
            elif isinstance(result, dict):
                sources.append({
                    "source": result.get('source', 'Unknown'),
                    "excerpt": result.get('content', '')[:200],
                    "confidence": result.get('score', 0.5),
                })
        return sources
    
    def calculate_confidence(self, search_results) -> float:
        if not search_results:
            return 0.3
        
        scores = []
        for result in search_results:
            if hasattr(result, 'score'):
                scores.append(result.score)
            elif isinstance(result, dict):
                scores.append(result.get('score', 0.5))
        
        return sum(scores) / len(scores) if scores else 0.5


__all__ = ['ExplainabilityEngine', 'ExplainedResponse']