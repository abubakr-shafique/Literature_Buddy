"""LLM/VLM backend abstractions and implementations + ModelManager.

Abstraction keeps the rest of the app provider-agnostic (spec §17).
ModelManager enforces hardware-aware residency: on the 16 GB profile the
LLM and VLM are loaded sequentially, never simultaneously.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Iterator, Protocol

from literature_buddy.config.settings import AppSettings


class LLMBackend(Protocol):
    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float) -> str: ...
    def stream(self, messages: list[dict], max_tokens: int,
               temperature: float) -> Iterator[str]: ...
    def unload(self) -> None: ...


class VLMBackend(Protocol):
    def analyze_image(self, prompt: str, image_path: str,
                      max_tokens: int = 1536) -> str: ...
    def unload(self) -> None: ...


# --------------------------- Ollama (default) ---------------------------

class OllamaLLMBackend:
    def __init__(self, model: str):
        from ollama import Client
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=host)
        self._model = model

    def generate(self, messages, max_tokens, temperature) -> str:
        r = self._client.chat(
            model=self._model, messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens})
        return r["message"]["content"]

    def stream(self, messages, max_tokens, temperature) -> Iterator[str]:
        for part in self._client.chat(
                model=self._model, messages=messages, stream=True,
                options={"temperature": temperature, "num_predict": max_tokens}):
            yield part["message"]["content"]

    def unload(self) -> None:
        # Ollama auto-unloads after keep_alive; explicit unload = keep_alive 0.
        self._client.generate(model=self._model, keep_alive=0)


class OllamaVLMBackend:
    def __init__(self, model: str, max_tokens: int = 1536):
        from ollama import Client
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=host)
        self._model = model
        self._max_tokens = max_tokens

    def analyze_image(self, prompt: str, image_path: str, max_tokens: int = 0) -> str:
        r = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt,
                       "images": [str(image_path)]}],
            options={"num_predict": max_tokens or self._max_tokens})
        return r["message"]["content"]

    def unload(self) -> None:
        self._client.generate(model=self._model, keep_alive=0)


# ------------------ llama.cpp (GGUF, tight VRAM control) ----------------

class LlamaCppLLMBackend:
    def __init__(self, model_path: str, n_gpu_layers: int = -1,
                 context_window: int = 32768):
        from llama_cpp import Llama
        self._llm = Llama(model_path=model_path, n_ctx=context_window,
                          n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, messages, max_tokens, temperature) -> str:
        r = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature)
        return r["choices"][0]["message"]["content"]

    def stream(self, messages, max_tokens, temperature) -> Iterator[str]:
        for tok in self._llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=True):
            if chunk := tok["choices"][0].get("delta", {}).get("content"):
                yield chunk

    def unload(self) -> None:
        del self._llm


# ----------------- Transformers (bitsandbytes 4-bit/8-bit) ---------------

class TransformersLLMBackend:
    def __init__(self, model_path: str, quantization: str = "4bit"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = None
        if quantization == "4bit" and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_compute_dtype=torch.float16)
        elif quantization == "8bit" and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(load_in_8bit=True)
        self._tok = AutoTokenizer.from_pretrained(model_path)
        kwargs = {"quantization_config": bnb} if bnb else {"torch_dtype": torch.float16}
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", **kwargs)

    def generate(self, messages, max_tokens, temperature) -> str:
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(
            self._model.device)
        out = self._model.generate(inputs, max_new_tokens=max_tokens,
                                   temperature=max(temperature, 1e-5), do_sample=True)
        return self._tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

    def stream(self, messages, max_tokens, temperature) -> Iterator[str]:
        # Non-streaming fallback keeps the Protocol; swap in TextIteratorStreamer if desired.
        yield self.generate(messages, max_tokens, temperature)

    def unload(self) -> None:
        import gc, torch
        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TransformersVLMBackend:
    """Transformers VLM, e.g. Qwen3-VL via AutoModelForVision2Seq [11]."""

    def __init__(self, model_path: str):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self._proc = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path, device_map="auto", dtype="auto")

    def analyze_image(self, prompt: str, image_path: str, max_tokens: int = 1536) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "url": f"file://{Path(image_path).resolve()}"},
            {"type": "text", "text": prompt}]}]
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=[str(image_path)],
                            return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=max_tokens)
        return self._proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def unload(self) -> None:
        import gc, torch
        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------- ModelManager ------------------------------

class ModelManager:
    """Lazily builds backends; on 16 GB profile enforces sequential residency."""

    def __init__(self, settings: AppSettings):
        self._s = settings
        self._llm: LLMBackend | None = None
        self._vlm: VLMBackend | None = None

    @property
    def sequential(self) -> bool:
        return self._s.profile == "16gb"  # 24 GB: co-residency allowed

    def llm(self) -> LLMBackend:
        if self._llm is None:
            cfg = self._s.llm
            if self.sequential and self._vlm is not None:
                self._vlm.unload(); self._vlm = None
            if cfg.provider == "ollama":
                self._llm = OllamaLLMBackend(cfg.model)
            elif cfg.provider == "llamacpp":
                self._llm = LlamaCppLLMBackend(cfg.model_path, cfg.n_gpu_layers,
                                               cfg.context_window)
            else:
                self._llm = TransformersLLMBackend(cfg.model_path, cfg.quantization)
        return self._llm

    def vlm(self) -> VLMBackend:
        if self._vlm is None:
            cfg = self._s.vlm
            if self.sequential and self._llm is not None:
                self._llm.unload(); self._llm = None
            if cfg.provider == "ollama":
                self._vlm = OllamaVLMBackend(cfg.model, cfg.max_tokens)
            else:
                self._vlm = TransformersVLMBackend(cfg.model_path)
        return self._vlm