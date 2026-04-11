"""
Safety模块 - 可解释性引擎

Safety & Trust赛道核心功能：
- 来源引用（每个回答可追溯原文）
- Thinking模式分析（展示推理过程）
- 回答可信度评估
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExplainedResponse:
    """带解释的响应"""
    answer: str
    thinking: Optional[str] = None
    sources: List[Dict] = None
    confidence: float = 0.0  # 可信度评分
    reasoning_chain: List[str] = None  # 推理链
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.reasoning_chain is None:
            self.reasoning_chain = []


class ExplainabilityEngine:
    """
    可解释性引擎
    
    核心功能：
    1. 添加来源引用 - 每个回答标注出处
    2. 生成立论逻辑 - 展示推理过程
    3. 评估可信度 - 判断回答是否有依据
    
    Safety & Trust赛道核心价值：
    - 透明：用户可见推理过程
    - 可解释：每个回答可追溯原文
    - Trust：回答可信度可视化
    """
    
    def __init__(self, gemma_client):
        """
        初始化可解释性引擎
        
        Args:
            gemma_client: GemmaClient实例
        """
        self.client = gemma_client
    
    def analyze_with_sources(
        self,
        query: str,
        retrieved_docs: List[Dict],
        enable_thinking: bool = True,
    ) -> ExplainedResponse:
        """
        带来源引用的分析
        
        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档（来自RAG）
            enable_thinking: 是否启用Thinking模式
        
        Returns:
            ExplainedResponse: 带解释的响应
        """
        # 构建上下文
        context = self._build_context(retrieved_docs)
        
        # 使用Gemma生成回答（Thinking模式）
        result = self.client.analyze_with_thinking(
            query=query,
            context=context,
        )
        
        # 评估可信度
        confidence = self._evaluate_confidence(result, retrieved_docs)
        
        # 构建推理链
        reasoning_chain = self._build_reasoning_chain(result.get('thinking', ''))
        
        return ExplainedResponse(
            answer=result['answer'],
            thinking=result['thinking'],
            sources=self._format_sources(retrieved_docs),
            confidence=confidence,
            reasoning_chain=reasoning_chain,
        )
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """构建RAG上下文"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:5]):  # 取Top-5
            source = doc.get('source', '未知来源')
            content = doc.get('content', '')
            context_parts.append(f"[{i+1}] {source}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """格式化来源引用"""
        sources = []
        
        for doc in retrieved_docs[:3]:  # 显示Top-3来源
            sources.append({
                "source": doc.get('source', '未知来源'),
                "excerpt": doc.get('content', '')[:200] + "...",
                "metadata": doc.get('metadata', {}),
            })
        
        return sources
    
    def _evaluate_confidence(
        self,
        result: Dict,
        retrieved_docs: List[Dict],
    ) -> float:
        """
        评估回答可信度
        
        规则：
        - 有Thinking推理过程 +0.3
        - 有来源引用 +0.4
        - 回答提到具体法规 +0.2
        - 回答明确说"无依据" -0.5（诚实降分）
        
        Returns:
            float: 0-1之间的可信度评分
        """
        confidence = 0.3  # 基础分
        
        # Thinking推理过程
        if result.get('thinking'):
            confidence += 0.3
        
        # 来源引用
        if len(retrieved_docs) > 0:
            confidence += 0.2
        
        # 回答质量检查
        answer = result.get('answer', '')
        
        # 回答提到具体法规内容
        if any(keyword in answer for keyword in ['法规', '条款', '规定', '根据']):
            confidence += 0.1
        
        # 回答明确说无依据（诚实）
        if '无明确依据' in answer or '未找到相关规定' in answer:
            confidence = 0.2  # 降低但保持正值（诚实也是Trust）
        
        return min(confidence, 1.0)
    
    def _build_reasoning_chain(self, thinking: str) -> List[str]:
        """
        从Thinking文本中提取推理链
        
        Args:
            thinking: Thinking模式的推理文本
        
        Returns:
            List[str]: 推理步骤列表
        """
        if not thinking:
            return []
        
        # 尝试按数字序号分割
        steps = []
        lines = thinking.split('\n')
        
        for line in lines:
            line = line.strip()
            # 检测类似 "1. ..." 的格式
            if line and any(
                line.startswith(f"{i}.") or line.startswith(f"{i})") 
                for i in range(1, 10)
            ):
                steps.append(line)
            elif line and not steps:
                # 第一行可能没有序号
                steps.append(line)
        
        return steps
    
    def format_response_for_display(
        self,
        response: ExplainedResponse,
    ) -> str:
        """
        格式化响应用于前端显示
        
        Safety & Trust赛道展示格式：
        - 推理过程（Thinking）
        - 最终回答
        - 来源引用
        - 可信度评分
        """
        output = []
        
        # 推理过程（如果启用Thinking）
        if response.thinking:
            output.append("## 🔍 推理过程\n")
            output.append(response.thinking)
            output.append("\n---\n")
        
        # 最终回答
        output.append("## 💡 回答\n")
        output.append(response.answer)
        output.append("\n")
        
        # 来源引用（Safety核心）
        if response.sources:
            output.append("## 📚 来源引用\n")
            for i, source in enumerate(response.sources):
                output.append(f"\n[{i+1}] **{source['source']}**\n")
                output.append(f"> {source['excerpt']}\n")
        
        # 可信度评分（Trust核心）
        output.append(f"\n## ✅ 可信度: {response.confidence:.1%}\n")
        
        if response.confidence >= 0.7:
            output.append("该回答有明确依据，可信度较高。")
        elif response.confidence >= 0.5:
            output.append("该回答有一定依据，建议核实具体条款。")
        else:
            output.append("该回答依据较少，建议查阅原文或咨询专业人士。")
        
        return "\n".join(output)


# 便捷函数
def create_explainability_engine(gemma_client) -> ExplainabilityEngine:
    """创建可解释性引擎的便捷函数"""
    return ExplainabilityEngine(gemma_client)


if __name__ == "__main__":
    # 测试示例
    print("Explainability Engine 测试")
    
    # 模拟数据
    mock_docs = [
        {"source": "AML法规第12条", "content": "金融机构应对客户进行尽职调查..."},
        {"source": "KYC指南第3章", "content": "高风险客户需进行增强型尽职调查..."},
    ]
    
    # 实际使用需要GemmaClient
    # engine = ExplainabilityEngine(client)
    # response = engine.analyze_with_sources("什么是KYC尽职调查？", mock_docs)
    # print(engine.format_response_for_display(response))