from typing import Sequence, Any, Optional, Dict, List
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import LLMMetadata, CompletionResponse, ChatMessage, ChatResponse
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.bridge.pydantic import PrivateAttr
import torch
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import MessageRole
from config import MIXED_TOP_K, MAX_NEW_TOKENS, CONTEXT_WINDOW, MODEL_DO_SAMPLE, MODEL_TEMPERATURE, MODEL_TOP_P, REPETITION_PENALTY, SYSTEM_TEMPLATE


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
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = MODEL_TEMPERATURE,
        top_p: float = MODEL_TOP_P,
        do_sample: bool = MODEL_DO_SAMPLE,
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
            eos_token_id=self._tokenizer.eos_token_id, #supposed to help stop cleanly, not in original and may remove if errors
            pad_token_id=self._tokenizer.pad_token_id, 
        )

    @property
    def metadata(self) -> LLMMetadata:
        ctx = CONTEXT_WINDOW

        return LLMMetadata(
            model_name="local-hf-seq2seq" if self._is_seq2seq else "local-hf-causal",
            context_window=ctx,
            num_output=self._gen_defaults["max_new_tokens"],
            is_chat_model=True, # if true, uses chat() instead of complete(). Phi better suited to chat so later fix
            is_function_calling_model=False,
        )

    def _apply_defaults(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        bad_keys = {"formatted"}
        clean = {k: v for k, v in kwargs.items() if k not in bad_keys}

        for k, v in self._gen_defaults.items():
            clean.setdefault(k, v)

        clean.setdefault("truncation", True)
        clean.setdefault("repetition_penalty", REPETITION_PENALTY) #added, can be removed if errors

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
            out = self._generate_seq2seq(prompt, params)
            self._debug_preview("seq2seq output", out)
            return out

        if self._generator is None:
            raise RuntimeError("Causal model generator was not initialized.")

        raw = self._generator(prompt, **params)

        try:
            text = raw[0]["generated_text"]
        except Exception as e:
            print("[DEBUG] failed to parse pipeline output:", e)
            raise

        return text


    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        params = self._apply_defaults(kwargs)

        if not self._is_seq2seq:
            hf_messages = [
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt},
            ]
            prompt = self._tokenizer.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True
            )

        out = self._generate_text(prompt, params)
        resp = CompletionResponse(text=out)
        return resp
    

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        params = self._apply_defaults(kwargs)
        full = self._generate_text(prompt, params)

        buf = ""
        for ch in full:
            buf += ch
            yield CompletionResponse(text=buf, delta=ch)


    def _format_chat(self, messages: Sequence[ChatMessage]) -> str:
        hf_messages = []

        # Check if system message already exists
        has_system = any(
            (m.role.value if hasattr(m.role, "value") else str(m.role)) == "system"
            for m in messages
        )

        # Inject system prompt if missing
        if not has_system:
            hf_messages.append({
                "role": "system",
                "content": SYSTEM_TEMPLATE
            })

        # Add all existing messages
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            hf_messages.append({
                "role": role,
                "content": m.content
            })

        prompt = self._tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt


    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._format_chat(messages)
        params = self._apply_defaults(kwargs)
        out = self._generate_text(prompt, params)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=out))


    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        prompt = self._format_chat(messages)
        params = self._apply_defaults(kwargs)
        full = self._generate_text(prompt, params)

        buf = ""
        for ch in full:
            buf += ch
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=buf),delta=ch,)


# -----------------------------
# Better hybrid retriever:
# uses Reciprocal Rank Fusion (RRF)
# -----------------------------
class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever using:
    - vector retrieval
    - BM25 retrieval
    - dedupe by node id
    - reciprocal rank fusion (RRF)

    This avoids unfairly favoring vector results and gives BM25
    a real chance to contribute before reranking.
    """
    def __init__(
        self,
        vec_retriever,
        bm25_retriever,
        final_top_k: int = MIXED_TOP_K,
        rrf_k: int = 60,
        debug: bool = True,
    ):
        super().__init__()
        self._vec = vec_retriever
        self._bm25 = bm25_retriever
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._debug = debug

    def _get_node_id(self, nws: NodeWithScore) -> str:
        node = getattr(nws, "node", nws)
        nid = (
            getattr(node, "node_id", None)
            or getattr(node, "id_", None)
            or getattr(nws, "id_", None)
        )

        if nid is None:
            text = getattr(node, "text", "") or (
                node.get_content() if hasattr(node, "get_content") else ""
            )
            nid = str(hash(text))

        return nid

    def _preview(self, nws: NodeWithScore, limit: int = 120) -> str:
        node = getattr(nws, "node", nws)
        text = getattr(node, "text", "") or (
            node.get_content() if hasattr(node, "get_content") else ""
        )
        return text[:limit].replace("\n", " ")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vec_nodes = self._vec.retrieve(query_bundle)
        bm25_nodes = self._bm25.retrieve(query_bundle)

        if self._debug:
            print("\nVECTOR RESULTS:")
            for i, n in enumerate(vec_nodes[:5]):
                print(i, n.score, self._preview(n))

            print("\nBM25 RESULTS:")
            for i, n in enumerate(bm25_nodes[:5]):
                print(i, n.score, self._preview(n))

        id_to_node: Dict[str, NodeWithScore] = {}
        fused_scores: Dict[str, float] = {}

        # Vector contribution
        for rank, nws in enumerate(vec_nodes, start=1):
            nid = self._get_node_id(nws)
            id_to_node[nid] = nws
            fused_scores[nid] = fused_scores.get(nid, 0.0) + 1.0 / (self._rrf_k + rank)

        # BM25 contribution
        for rank, nws in enumerate(bm25_nodes, start=1):
            nid = self._get_node_id(nws)
            if nid not in id_to_node:
                id_to_node[nid] = nws
            fused_scores[nid] = fused_scores.get(nid, 0.0) + 1.0 / (self._rrf_k + rank)

        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda nid: fused_scores[nid],
            reverse=True,
        )

        results: List[NodeWithScore] = []
        for nid in ranked_ids[: self._final_top_k]:
            nws = id_to_node[nid]
            results.append(NodeWithScore(node=nws.node, score=fused_scores[nid]))

        if self._debug:
            print("\nFUSED RESULTS:")
            for i, n in enumerate(results[:10]):
                print(i, n.score, self._preview(n))

        return results