"""Gemma 4 Client - Multi-backend LLM inference"""

import os
from typing import Optional, Dict, Any, List
from enum import Enum

from ..data.models import GemmaResponse


class BackendType(Enum):
    KAGGLE = "kaggle"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class GemmaClient:
    DEFAULT_MODEL = "gemma-4-26b-a4b"
    KAGGLE_MODEL_PATH = "google/gemma-4/transformers/gemma-4-26b-a4b"
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
        self.backend = BackendType(backend.lower())
        self.model = model or self.DEFAULT_MODEL
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        self._init_backend(**kwargs)
    
    def _init_backend(self, **kwargs):
        if self.backend == BackendType.KAGGLE:
            self._init_kaggle_backend(**kwargs)
        elif self.backend == BackendType.OLLAMA:
            self._init_ollama_backend(**kwargs)
        elif self.backend == BackendType.HUGGINGFACE:
            self._init_hf_backend(**kwargs)
    
    def _init_kaggle_backend(self, **kwargs):
        try:
            import kagglehub
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            
            model_path = kagglehub.model_download(self.KAGGLE_MODEL_PATH)
            
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=self.device_map,
                **kwargs
            )
            
        except ImportError as e:
            raise ImportError("Install: kagglehub, transformers, torch, bitsandbytes") from e
    
    def _init_ollama_backend(self, **kwargs):
        try:
            import ollama
            
            self._ollama = ollama
            
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model not in model_names:
                ollama.pull(self.model)
            
        except ImportError as e:
            raise ImportError("Install: ollama") from e
    
    def _init_hf_backend(self, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            
            model_path = kwargs.get('model_path', f"google/{self.model}")
            
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=self.device_map,
                **kwargs
            )
            
        except ImportError as e:
            raise ImportError("Install: transformers, torch, bitsandbytes") from e
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: str = "You are an AML compliance expert.",
    ) -> GemmaResponse:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{context}"
            })
        
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature or self.DEFAULT_TEMPERATURE
        tp = top_p or self.DEFAULT_TOP_P
        tk = top_k or self.DEFAULT_TOP_K
        
        if self.backend in (BackendType.KAGGLE, BackendType.HUGGINGFACE):
            return self._generate_transformers(
                messages, enable_thinking, max_new_tokens, temp, tp, tk
            )
        elif self.backend == BackendType.OLLAMA:
            return self._generate_ollama(
                messages, enable_thinking, max_new_tokens, temp
            )
    
    def analyze_with_thinking(
        self,
        query: str,
        context: str = "",
        enable_thinking: bool = True,
    ) -> Dict[str, Any]:
        """Analyze a query with thinking mode enabled. Returns a dict with answer and thinking."""
        response = self.generate(
            prompt=query,
            context=context,
            enable_thinking=enable_thinking,
        )
        return {
            "answer": response.content,
            "thinking": response.thinking,
        }
    
    def _generate_transformers(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> GemmaResponse:
        import torch
        
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
        
        raw_response = self._processor.decode(
            outputs[0][input_len:], 
            skip_special_tokens=False
        )
        
        return self._parse_response(raw_response, enable_thinking)
    
    def _generate_ollama(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> GemmaResponse:
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": max_new_tokens,
                "temperature": temperature,
            }
        )
        
        content = response.get('message', {}).get('content', '')
        
        return GemmaResponse(
            thinking=None,
            content=content,
            raw_response=content,
        )
    
    def _parse_response(
        self, 
        raw_response: str, 
        enable_thinking: bool
    ) -> GemmaResponse:
        thinking = None
        content = raw_response
        
        if enable_thinking and "<|think|>" in raw_response:
            think_start = raw_response.find("<|think|>") + len("<|think|>")
            think_end = raw_response.find("<|/think|>")
            
            if think_end > think_start:
                thinking = raw_response[think_start:think_end].strip()
                content = raw_response[think_end + len("<|/think|>"):].strip()
        
        content = content.replace("<|end|>", "").strip()
        
        return GemmaResponse(
            thinking=thinking,
            content=content,
            raw_response=raw_response,
        )


__all__ = ['GemmaClient', 'BackendType']