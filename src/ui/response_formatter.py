"""Response formatter"""

from typing import Dict, Any, List
from ..data.models import GemmaResponse


class ResponseFormatter:
    def format_sources(self, sources: List[Dict]) -> List[Dict]:
        return [
            {
                "source": s.get("source", "Unknown"),
                "page": s.get("page", "-"),
                "excerpt": s.get("excerpt", "")[:200] + "..." if len(s.get("excerpt", "")) > 200 else s.get("excerpt", ""),
                "confidence": s.get("confidence", 0.0),
            }
            for s in sources
        ]
    
    def format_answer(self, response: GemmaResponse) -> Dict[str, Any]:
        return {
            "content": response.content,
            "thinking": response.thinking,
            "sources": self.format_sources(response.sources),
            "has_sources": len(response.sources) > 0,
        }
    
    def format_for_streamlit(self, response: GemmaResponse) -> str:
        output = f"### Answer\n\n{response.content}\n\n"
        
        if response.thinking:
            output += f"### Reasoning\n\n{response.thinking}\n\n"
        
        if response.sources:
            output += "### Sources\n\n"
            for i, source in enumerate(response.sources, 1):
                output += f"{i}. **{source.get('source')}** (p.{source.get('page', '-')})\n"
                output += f"   > {source.get('excerpt', '')[:100]}...\n\n"
        
        return output


__all__ = ['ResponseFormatter']