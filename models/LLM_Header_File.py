from typing import Sequence, Any, Optional, Dict
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import LLMMetadata, CompletionResponse, ChatMessage, ChatResponse
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.bridge.pydantic import PrivateAttr
import torch

class HuggingFaceLLM(CustomLLM):
    """
    Works with BOTH causal LMs (e.g., distilgpt2) and seq2seq LMs (e.g., google/flan-t5-base).
    Auto-detects model type and builds the correct HF pipeline.

    Key behaviors:
    - Sets pad_token_id if missing.
    - For causal LMs, sets return_full_text=False to avoid echoing the prompt.
    - Exposes generation params via __init__.
    - Uses tokenizer.model_max_length for accurate context window.
    """
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _generator: Any = PrivateAttr()
    _is_seq2seq: bool = PrivateAttr()
    _gen_defaults: Dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        device: Optional[int] = None,   # None = auto, int = cuda device id, -1 = CPU
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        from transformers import (
            AutoTokenizer,
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            pipeline,
        )

        # Load tokenizer + config first to decide which model class to use
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self._tokenizer.model_max_length = 450     # or any context window you want
        self._tokenizer.truncation_side = "right"
        config = AutoConfig.from_pretrained(model_path)

        # Decide if encoder-decoder (seq2seq) or decoder-only (causal)
        self._is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))

        if self._is_seq2seq:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self._generator = None  # we'll call model.generate() directly
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_path)
            task = "text-generation"
            # Set pad_token_id to eos if missing to avoid warnings/blanks
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._generator = pipeline(
                task, model=self._model, tokenizer=self._tokenizer, device=device
            )

        # Reasonable generation defaults; can be overridden per-call
        self._gen_defaults = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    @property
    def metadata(self) -> LLMMetadata:
        # Use tokenizer's real window; many HF tokenizers use large sentinel values, guard it
        ctx = getattr(self._tokenizer, "model_max_length", 2048)
        if ctx and ctx > 100_000_000_000:  # HF sometimes sets very large int for "infinite"
            ctx = 2048
        return LLMMetadata(
            model_name="local-hf-seq2seq" if self._is_seq2seq else "local-hf-causal",
            context_window=ctx,
            num_output=self._gen_defaults["max_new_tokens"],
            is_chat_model=False,
            is_function_calling_model=False,
        )

    def _apply_defaults(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # LlamaIndex passes extra args; ignore ones the HF pipeline doesn't know
        bad_keys = {"formatted"}  # common from LI
        clean = {k: v for k, v in kwargs.items() if k not in bad_keys}
        # Our defaults only fill missing values
        for k, v in self._gen_defaults.items():
            clean.setdefault(k, v)
        # For causal models, avoid echoing prompt
        if not self._is_seq2seq:
            clean.setdefault("return_full_text", False)
            clean.setdefault("truncation", True)
        else:
            # text2text-generation handles truncation internally; keep it explicit anyway
            clean.setdefault("truncation", True)
        return clean

    # ---------- Completion APIs ----------

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        params = self._apply_defaults(kwargs)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._tokenizer.model_max_length,
        )

        # Move tensors to model device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=params.get("max_new_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            do_sample=params.get("do_sample"),
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        out = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return CompletionResponse(text=out)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        # Simulated streaming by chunking the final text (HF pipeline doesn't stream)
        params = self._apply_defaults(kwargs)
        full = self._generator(prompt, **params)[0]["generated_text"]
        buf = ""
        for ch in full:
            buf += ch
            yield CompletionResponse(text=buf, delta=ch)

    # ---------- Chat APIs ----------

    def _format_chat(self, messages: Sequence[ChatMessage]) -> str:
        """
        Tiny, neutral chat template that works for both causal and seq2seq models.
        You can swap this for a model-specific template if needed.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        user_parts = [m.content for m in messages if m.role == "user"]
        assistant_parts = [m.content for m in messages if m.role == "assistant"]

        system = f"System: {system_parts[-1]}\n" if system_parts else ""
        history = ""
        for m in messages[:-1]:
            if m.role == "user":
                history += f"User: {m.content}\n"
            elif m.role == "assistant":
                history += f"Assistant: {m.content}\n"
        # last message should be user
        last = messages[-1].content if messages else ""
        prompt = f"{system}{history}User: {last}\nAssistant:"
        return prompt

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._format_chat(messages)
        params = self._apply_defaults(kwargs)
        out = self._generator(prompt, **params)[0]["generated_text"]
        return ChatResponse(content=out)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        prompt = self._format_chat(messages)
        params = self._apply_defaults(kwargs)
        full = self._generator(prompt, **params)[0]["generated_text"]
        buf = ""
        for ch in full:
            buf += ch
            yield ChatResponse(content=buf, delta=ch)
