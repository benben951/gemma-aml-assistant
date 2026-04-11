"""
Gemma 4 Client - Multi-backend LLM inference client

支持多种推理框架:
- Kaggle Notebook (免费GPU)
- Ollama (本地部署)
- Hugging Face transformers

默认使用 26B A4B 模型（活跃参数仅3.8B）
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    KAGGLE = "kaggle"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class GemmaResponse:
    """Gemma 响应结果"""
    content: str  # 最终回答内容（必填）
    raw_response: str  # 原始响应（必填）
    thinking: Optional[str] = None  # Thinking模式的推理过程
    sources: List[Dict] = None  # 来源引用（RAG场景）
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class GemmaClient:
    """
    Gemma 4 推理客户端
    
    示例:
        client = GemmaClient(backend="kaggle")
        response = client.generate("什么是AML尽职调查？", enable_thinking=True)
        print(response.thinking)  # 推理过程
        print(response.content)   # 最终回答
    """
    
    # 默认模型配置
    DEFAULT_MODEL = "gemma-4-26b-a4b"  # 26B A4B (活跃参数3.8B)
    KAGGLE_MODEL_PATH = "google/gemma-4/transformers/gemma-4-26b-a4b"
    
    # 推荐采样参数
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TOP_P = 0.95
    DEFAULT_TOP_K = 64
    
    def __init__(
        self,
        backend: str = "kaggle",
        model: Optional[str] = None,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        **kwargs
    ):
        """
        初始化 Gemma 客户端
        
        Args:
            backend: 推理框架类型 (kaggle/ollama/huggingface)
            model: 模型名称/路径
            device_map: 设备映射策略
            load_in_4bit: 是否使用4-bit量化
        """
        self.backend = BackendType(backend.lower())
        self.model = model or self.DEFAULT_MODEL
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        # 初始化对应的backend
        self._init_backend(**kwargs)
    
    def _init_backend(self, **kwargs):
        """初始化指定的推理框架"""
        if self.backend == BackendType.KAGGLE:
            self._init_kaggle_backend(**kwargs)
        elif self.backend == BackendType.OLLAMA:
            self._init_ollama_backend(**kwargs)
        elif self.backend == BackendType.HUGGINGFACE:
            self._init_hf_backend(**kwargs)
    
    def _init_kaggle_backend(self, **kwargs):
        """初始化 Kaggle Notebook backend"""
        try:
            import kagglehub
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            
            # 下载模型（Kaggle环境会自动缓存）
            model_path = kagglehub.model_download(self.KAGGLE_MODEL_PATH)
            
            # 4-bit量化配置
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            # 加载模型
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=self.device_map,
                **kwargs
            )
            
            print(f"Gemma 4 模型加载成功: {self.model}")
            print(f"Backend: Kaggle, 量化: {self.load_in_4bit}")
            
        except ImportError as e:
            raise ImportError(
                "Kaggle backend 需要安装: kagglehub, transformers, torch, bitsandbytes"
            ) from e
    
    def _init_ollama_backend(self, **kwargs):
        """初始化 Ollama backend"""
        try:
            import ollama
            
            # Ollama 不需要显式加载模型，调用时会自动拉取
            self._ollama = ollama
            
            # 检查模型是否已下载
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model not in model_names:
                print(f"模型 {self.model} 未下载，正在拉取...")
                ollama.pull(self.model)
            
            print(f"Ollama backend 初始化成功: {self.model}")
            
        except ImportError as e:
            raise ImportError("Ollama backend 需要安装: ollama") from e
    
    def _init_hf_backend(self, **kwargs):
        """初始化 Hugging Face backend"""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            
            model_path = kwargs.get('model_path', f"google/{self.model}")
            
            # 4-bit量化配置
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            # 加载模型
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=self.device_map,
                **kwargs
            )
            
            print(f"Hugging Face backend 初始化成功: {model_path}")
            
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend 需要安装: transformers, torch, bitsandbytes"
            ) from e
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: str = "你是一个AML合规专家。请基于法规内容回答问题。",
    ) -> GemmaResponse:
        """
        生成回答
        
        Args:
            prompt: 用户问题
            context: RAG上下文（可选）
            enable_thinking: 是否启用Thinking模式
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: Top-p采样参数
            top_k: Top-k采样参数
            system_prompt: 系统提示
        
        Returns:
            GemmaResponse: 包含thinking和content的响应
        """
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # 添加上下文（RAG场景）
        if context:
            messages.append({
                "role": "system",
                "content": f"以下是与问题相关的法规内容：\n{context}"
            })
        
        messages.append({"role": "user", "content": prompt})
        
        # 采样参数
        temp = temperature or self.DEFAULT_TEMPERATURE
        tp = top_p or self.DEFAULT_TOP_P
        tk = top_k or self.DEFAULT_TOP_K
        
        # 根据backend生成响应
        if self.backend == BackendType.KAGGLE or self.backend == BackendType.HUGGINGFACE:
            return self._generate_transformers(
                messages, enable_thinking, max_new_tokens, temp, tp, tk
            )
        elif self.backend == BackendType.OLLAMA:
            return self._generate_ollama(
                messages, enable_thinking, max_new_tokens, temp
            )
    
    def _generate_transformers(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> GemmaResponse:
        """使用 transformers 生成响应"""
        import torch
        
        # 应用聊天模板
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # 关键！启用Thinking模式
        )
        
        # 编码输入
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # 生成
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
        
        # 解码响应
        raw_response = self._processor.decode(
            outputs[0][input_len:], 
            skip_special_tokens=False
        )
        
        # 解析Thinking结果
        return self._parse_response(raw_response, enable_thinking)
    
    def _generate_ollama(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> GemmaResponse:
        """使用 Ollama 生成响应"""
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": max_new_tokens,
                "temperature": temperature,
            }
        )
        
        content = response.get('message', {}).get('content', '')
        
        # Ollama 目前不支持显式Thinking模式解析
        return GemmaResponse(
            thinking=None if not enable_thinking else "Ollama暂不支持Thinking模式解析",
            content=content,
            raw_response=content,
        )
    
    def _parse_response(
        self, 
        raw_response: str, 
        enable_thinking: bool
    ) -> GemmaResponse:
        """
        解析响应，提取Thinking和Content
        
        Gemma 4 Thinking模式格式:
        <|think|>
        推理过程...
        <|/think|>
        最终回答...
        """
        thinking = None
        content = raw_response
        
        if enable_thinking and "<|think|>" in raw_response:
            # 提取Thinking部分
            think_start = raw_response.find("<|think|>") + len("<|think|>")
            think_end = raw_response.find("<|/think|>")
            
            if think_end > think_start:
                thinking = raw_response[think_start:think_end].strip()
                content = raw_response[think_end + len("<|/think|>"):].strip()
        
        # 清理特殊token
        content = content.replace("<|end|>", "").strip()
        
        return GemmaResponse(
            thinking=thinking,
            content=content,
            raw_response=raw_response,
        )
    
    def analyze_with_thinking(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用Thinking模式进行合规分析（Safety & Trust赛道核心）
        
        Args:
            query: 合规问题
            context: AML法规上下文
        
        Returns:
            包含thinking、answer、sources的分析结果
        """
        response = self.generate(
            prompt=query,
            context=context,
            enable_thinking=True,
            system_prompt="你是一个AML合规专家。请逐步分析问题，给出详细推理过程。",
        )
        
        return {
            "thinking": response.thinking,
            "answer": response.content,
            "sources": [],  # RAG模块会填充
            "raw": response.raw_response,
        }


# 便捷函数
def create_gemma_client(backend: str = "kaggle", **kwargs) -> GemmaClient:
    """创建 Gemma 客户端的便捷函数"""
    return GemmaClient(backend=backend, **kwargs)


if __name__ == "__main__":
    # 测试示例
    print("Gemma 4 Client 测试")
    
    # 注意：实际运行需要安装依赖
    # client = GemmaClient(backend="ollama")
    # response = client.generate("什么是AML尽职调查？", enable_thinking=True)
    # print(f"Thinking: {response.thinking}")
    # print(f"Answer: {response.content}")