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

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self._tokenizer.model_max_length = 450
        self._tokenizer.truncation_side = "right"

        config = AutoConfig.from_pretrained(model_path)
        self._is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))

        if self._is_seq2seq:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self._generator = None
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_path)

            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            self._generator = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
            )

        self._gen_defaults = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    @property
    def metadata(self) -> LLMMetadata:
        ctx = getattr(self._tokenizer, "model_max_length", 2048)
        if ctx and ctx > 100_000_000_000:
            ctx = 2048

        return LLMMetadata(
            model_name="local-hf-seq2seq" if self._is_seq2seq else "local-hf-causal",
            context_window=ctx,
            num_output=self._gen_defaults["max_new_tokens"],
            is_chat_model=False,
            is_function_calling_model=False,
        )

    def _apply_defaults(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        bad_keys = {"formatted"}
        clean = {k: v for k, v in kwargs.items() if k not in bad_keys}

        for k, v in self._gen_defaults.items():
            clean.setdefault(k, v)

        clean.setdefault("truncation", True)

        if not self._is_seq2seq:
            clean.setdefault("return_full_text", False)

        return clean

    def _generate_seq2seq(self, prompt: str, params: Dict[str, Any]) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._tokenizer.model_max_length,
        )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=params.get("max_new_tokens"),
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            do_sample=params.get("do_sample"),
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _generate_text(self, prompt: str, params: Dict[str, Any]) -> str:
        if self._is_seq2seq:
            return self._generate_seq2seq(prompt, params)

        if self._generator is None:
            raise RuntimeError("Causal model generator was not initialized.")

        return self._generator(prompt, **params)[0]["generated_text"]

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        params = self._apply_defaults(kwargs)
        out = self._generate_text(prompt, params)
        return CompletionResponse(text=out)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        params = self._apply_defaults(kwargs)
        full = self._generate_text(prompt, params)

        buf = ""
        for ch in full:
            buf += ch
            yield CompletionResponse(text=buf, delta=ch)

    def _format_chat(self, messages: Sequence[ChatMessage]) -> str:
        system_parts = [m.content for m in messages if m.role == "system"]

        system = f"System: {system_parts[-1]}\n" if system_parts else ""
        history = ""

        for m in messages[:-1]:
            if m.role == "user":
                history += f"User: {m.content}\n"
            elif m.role == "assistant":
                history += f"Assistant: {m.content}\n"

        last = messages[-1].content if messages else ""
        prompt = f"{system}{history}User: {last}\nAssistant:"
        return prompt

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._format_chat(messages)
        params = self._apply_defaults(kwargs)
        out = self._generate_text(prompt, params)
        return ChatResponse(content=out)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        prompt = self._format_chat(messages)
        params = self._apply_defaults(kwargs)
        full = self._generate_text(prompt, params)

        buf = ""
        for ch in full:
            buf += ch
            yield ChatResponse(content=buf, delta=ch)