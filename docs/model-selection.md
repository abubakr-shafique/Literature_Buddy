# Model selection

_Verified against public sources on 2026-09-29. Model releases move fast: re-check model cards and `ollama.com/library` before you download. Items marked † rely on secondary sources (news/community listings) rather than the vendor page._

## Defaults

| Role | 16 GB | 24 GB | Notes |
|---|---|---|---|
| Chat + figures | **Qwen3.5-9B** (`Qwen/Qwen3.5-9B`, Ollama `qwen3.5:9b`†) | **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`, released 2026-08-14, Apache-2.0, native vision-language, 262K native context; Ollama `qwen3.8:27b`†) | Qwen 3.5+ are natively multimodal, so a single model serves text and images |
| Embeddings | Qwen3-Embedding-0.6B (Apache-2.0, 1024-d, instruction-aware) | same (or `-4B` if you have spare VRAM/CPU) | Ollama: `qwen3-embedding:0.6b` |
| Reranker | `BAAI/bge-reranker-v2-m3` (Apache-2.0, works with sentence-transformers `CrossEncoder`) | same | Qwen3-Reranker is an alternative but needs custom scoring code |

## Alternatives worth trying

| Model | Size / memory | Why |
|---|---|---|
| **Gemma 4 12B** (Apache-2.0, multimodal, 256K ctx) | Google lists ~6.7 GB at 4-bit, 13.4 GB at 8-bit | Strong all-rounder for 16 GB; the `lite` profile default |
| **Gemma 4 26B-A4B** (MoE) | ~14.4 GB at 4-bit | Fast (≈4B active parameters) with larger-model quality |
| **Gemma 4 31B** | ~17.5 GB at 4-bit | Dense option for 24 GB |
| Qwen3.6-27B / 35B-A3B | ~17 GB (Q4)† | Previous generation; well-supported in tooling |
| BGE-M3 (MIT), EmbeddingGemma-300M | ~1.1 GB / small | Embedding alternatives (CPU-friendly) |

Not evaluated here: InternVL, MiniCPM-V, medical/biology-specific VLMs (e.g. MedGemma). If your papers are dominated by microscopy, radiology or dense charts, benchmark a domain VLM on your own figures: set `vlm.same_as_llm: false` and give `vlm` its own provider/model (`ModelManager` swaps in-process models sequentially).

## Choosing quantisation

* Transformers + bitsandbytes NF4 (`quantization: 4bit`): smallest, good quality, default.
* 8-bit (`8bit`): better fidelity, ~2× memory, slower with bitsandbytes.
* Ollama/llama.cpp GGUF Q4_K_M/Q5/Q8 are typically faster than bitsandbytes; prefer the `-ollama` profiles when speed matters.

## VRAM budgeting (rule of thumb)

`weights + KV cache (grows with llm.context_length) + ~1.5–3 GB` for embedder/reranker on GPU. Keep `retrieval.max_context_tokens` well below `llm.context_length` (default 5.5k of 16k). Set `embedding.device: cpu` to free ~1–2 GB.

## Local weights & offline use

`python scripts/download_models.py --profile 16gb` stores repos as `./models/<org>__<name>`; they are loaded from there automatically (or set `llm.model_path`). Combine with `LB_OFFLINE=true`.

## Known caveat

Models with hybrid linear-attention layers (Qwen 3.5+ family) may run slowly under plain Transformers unless their optional fast-path kernels are installed (see the model card). This was not benchmarked here; if generation is slow, use the Ollama profile.
