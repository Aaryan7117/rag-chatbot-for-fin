"""
LLM Engine - Grounded Generation
=================================

LLM for generating responses based ONLY on retrieved context.
Uses LLaMA-3-8B-Instruct with 4-bit quantization.

Critical Design Principles:
- LLM is a REASONING engine, not a KNOWLEDGE source
- Temperature = 0 for deterministic output
- System prompt enforces grounded responses
- All answers must cite sources
- No speculation or external knowledge
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

# Lazy imports for heavy dependencies
_transformers_imported = False
_model = None
_tokenizer = None


def _import_transformers():
    """Lazy import of transformers and torch."""
    global _transformers_imported
    if _transformers_imported:
        return
    
    global torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline
    )
    
    _transformers_imported = True


# System prompt that enforces grounded responses
SYSTEM_PROMPT = """You are a financial document analyst assistant. Your role is to answer questions based ONLY on the provided context from SEC filings and financial documents.

CRITICAL RULES:
1. Use ONLY the information provided in the context below
2. Do NOT use any prior knowledge or external information
3. If the answer is not in the context, respond exactly with: "This information is not present in the provided documents."
4. Always cite your sources using the format [Section: X, Page Y]
5. Use neutral, analytical language
6. Be precise with numbers and financial terms
7. Do not speculate or make assumptions
8. If information is partial or unclear, acknowledge the limitation

You are a reasoning engine, not a knowledge source. Ground every statement in the provided context."""


class LLMEngine:
    """
    LLM engine for grounded generation.
    
    Uses LLaMA-3-8B-Instruct with 4-bit quantization for
    memory-efficient inference on 16GB GPUs.
    
    All responses are grounded in provided context.
    No external knowledge or speculation.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0
    ):
        """
        Initialize LLM engine.
        
        Args:
            model_name: HuggingFace model name
            quantization: "4bit", "8bit", or "none"
            device: "cuda" or "cpu"
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model_name = model_name or settings.models.llm_model
        self.quantization = quantization or settings.models.llm_quantization
        self.device = device or settings.models.device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Load model with quantization if not already loaded."""
        if self._pipeline is not None:
            return
        
        _import_transformers()
        
        print(f"[LLMEngine] Loading model: {self.model_name}")
        print(f"[LLMEngine] Quantization: {self.quantization}")
        print(f"[LLMEngine] Device: {self.device}")
        
        # Quantization config for 4-bit
        quantization_config = None
        if self.quantization == "4bit" and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit" and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if self.device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Create pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            return_full_text=False
        )
        
        print("[LLMEngine] Model loaded successfully")
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response grounded in the provided context.
        
        Args:
            query: User question
            context: Retrieved context with citations
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated response text
        """
        self._load_model()
        
        # Build prompt
        prompt = self._build_prompt(
            query=query,
            context=context,
            system_prompt=system_prompt or SYSTEM_PROMPT
        )
        
        # Generate
        outputs = self._pipeline(prompt)
        
        response = outputs[0]["generated_text"].strip()
        
        return response
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: str
    ) -> str:
        """
        Build prompt in chat format.
        
        Uses the LLaMA chat template format.
        """
        # LLaMA-3 chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context from documents:
{context}

Question: {query}

Please answer based ONLY on the context provided above. Include citations in the format [Section: X, Page Y]."""}
        ]
        
        # Apply chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback format
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

Context from documents:
{context}

Question: {query}

Please answer based ONLY on the context provided above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def generate_simple(self, prompt: str) -> str:
        """Generate from a raw prompt without template."""
        self._load_model()
        outputs = self._pipeline(prompt)
        return outputs[0]["generated_text"].strip()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._pipeline is not None


# Fallback for when transformers isn't available or model can't load
class SimpleLLMEngine:
    """
    Fallback LLM engine that uses simple extraction.
    
    Used when:
    - Transformers not installed
    - Model loading fails
    - Testing without GPU
    
    Extracts relevant sentences from context rather than generating.
    """
    
    def __init__(self):
        print("[SimpleLLMEngine] Using fallback extraction mode")
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Extract relevant sentences from context."""
        # Simple keyword matching
        query_words = set(query.lower().split())
        
        sentences = context.replace('\n', ' ').split('. ')
        
        # Score sentences by keyword overlap
        scored = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            if overlap > 0:
                scored.append((overlap, sent))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Take top sentences
        if scored:
            response = ". ".join(s[1] for s in scored[:3])
            return f"Based on the documents: {response}."
        else:
            return "This information is not present in the provided documents."
    
    @property
    def is_loaded(self) -> bool:
        return True


def get_llm_engine() -> LLMEngine:
    """Get or create the LLM engine singleton."""
    global _model
    
    if _model is None:
        try:
            _model = LLMEngine()
        except Exception as e:
            print(f"[LLMEngine] Failed to initialize: {e}")
            print("[LLMEngine] Falling back to simple extraction")
            _model = SimpleLLMEngine()
    
    return _model
