# Literature Buddy

**An interactive scientific literature reading assistant with local LLM/VLM support.**

Literature Buddy helps researchers read, understand, analyze, and discuss scientific papers. It combines a PDF viewer, scientific-document-aware parsing, multimodal retrieval-augmented generation (RAG), and an evidence-grounded chat interface — running on your own GPU with open-weight models.

> Not a generic "chat with PDF" tool — built specifically for scientific literature.

## Features

- **Scientific PDF processing** — section-aware parsing (abstract, methods, results...), figure/caption association, table extraction, references, page-level metadata with PyMuPDF
- **Figure understanding** — extract figures with captions; ask "what does Figure 3 show?" and get a VLM-grounded answer
- **Multimodal retrieval** — retrieve text chunks, figures, and tables together; hybrid dense + BM25 search with cross-encoder reranking
- **Evidence-grounded answers** — every answer cites page/section/figure numbers; clickable citations navigate the PDF viewer
- **Anti-hallucination design** — answers explicitly separate `[SOURCE: Paper]` (stated), `[SOURCE: Inference]` (derived), and `[SOURCE: General scientific knowledge]`
- **Local-first** — works offline after model download; Ollama, llama.cpp, and Transformers backends
- **Hardware-aware** — tuned profiles for 16 GB and 24 GB NVIDIA GPUs, 4-bit/8-bit quantization
- **Persistent indexes** — papers are processed once; embeddings and extractions are cached and reused
- **Professional desktop GUI** — PySide6/Qt, two-panel layout, background worker threads (GUI never freezes)

## Screenshots

```
┌──────────────────────────────────────────────────────────────┐
│ Literature Buddy                                             │
├─────────────────────────────┬────────────────────────────────┤
│                             │  Chat                          │
│        PDF                  │  You: What is the main         │
│        Viewer               │  contribution?                 │
│   ┌───────────────┐         │                                │
│   │   page 4/12   │         │  Assistant: The authors        │
│   │               │         │  propose an attention-based    │
│   │   [Figure 2]  │         │  MIL method [SOURCE: Paper]    │
│   │               │         │  ▸ Abstract, p.1               │
│   └───────────────┘         │  ▸ Results, p.6                │
│                             │                                │
│  ◀  Page 4 of 12  ▶  🔍    │  [ Ask about this paper... ]   │
└─────────────────────────────┴────────────────────────────────┘
```

## Supported models

| Role | Recommended | Alternatives |
|---|---|---|
| LLM (16 GB) | Qwen3-14B (Q4) | Llama-3.1-8B, Qwen3-8B |
| LLM (24 GB) | Qwen3-32B (Q4) | Qwen3-30B-A3B, gpt-oss-20b |
| VLM | Qwen3-VL-8B (Q4) | Qwen3-VL-32B (24 GB), Gemma-3-27B |
| Embeddings | BGE-M3 | SPECTER2 (paper-level), Qwen3-Embedding |
| Reranker | BGE-Reranker-v2-M3 | Qwen3-Reranker-0.6B/8B |

See [docs/model-selection.md](docs/model-selection.md) for full analysis.

## Hardware requirements

- GPU: NVIDIA 16 GB (minimum, 4-bit quant) or 24 GB (recommended)
- RAM: 32 GB recommended (16 GB workable)
- Disk: ~15–25 GB for models + caches
- OS: Linux/Windows/macOS — Linux+CUDA recommended
- Python: 3.11+

## Installation

```bash
# 1. Clone and set up environment
git clone https://github.com/abubakr-shafique/Literature_Buddy.git
cd literature-buddy
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows

# 2. Install
pip install -e ".[all]"

# 3. Install Ollama (default inference backend)
curl -fsSL https://ollama.com/install.sh | sh        # Linux
ollama pull qwen3:14b                                # LLM (~9 GB)
ollama pull qwen3-vl:8b                              # VLM (~6 GB)
```

## Model setup

Model providers are configured in `config/settings.yaml` (LLM, VLM, embeddings, quantization). The default stack works entirely through Ollama. To use Transformers/llama.cpp backends instead, set `llm.provider: transformers` or `llamacpp` and point to a local model path.

## Running

```bash
literature-buddy          # or: python -m literature_buddy
```

Then: **File → Open Paper** (local PDF or open-access URL) → wait for indexing → ask questions in the chat panel. Click a citation to jump to that page.

## Example usage

- "What is the main contribution of this paper?"
- "What dataset did they use and what were the inclusion criteria?"
- "Explain Figure 3 to me."
- "Which method performed best in Table 2?"
- "Do you see any limitations in their experimental design?" (answers distinguish author-stated limitations from assistant inference)

## Configuration

All behavior is controlled by `config/settings.yaml` and environment variables (see `.env.example`). Tunables: model names/paths, quantization, retrieval `top_k`, reranking, OCR, figure/table extraction, context budget.

## Architecture

PDF/URL loader → scientific parser (sections, figures, captions, tables, refs) → structure-aware chunker → embeddings (BGE-M3) + BM25 → persistent ChromaDB index → hybrid retrieval + reranking → context builder (text + figure images) → LLM/VLM (Ollama/llama.cpp/Transformers) → grounded answer + citations → Qt GUI.

See [docs/architecture.md](docs/architecture.md) for the full diagram and rationale.

## Development setup

```bash
pip install -e ".[dev]"
pytest tests/ -q
ruff check src/ && ruff format src/
```

## Testing

`tests/` covers document parsing, chunking, retrieval, context building, and citation formatting. LLM/VLM calls are mocked in unit tests (`tests/test_rag_pipeline.py`).

## Troubleshooting

- **Ollama not found** — start it with `ollama serve`, or set `LITERATURE_BUDDY_API_BASE`
- **CUDA OOM** — lower `llm.max_tokens`, switch to a smaller model, or increase `retrieval.top_k` discipline; see hardware profiles below
- **Slow first run** — embeddings model downloads once (~2 GB); subsequent runs are cached
- **Figures not detected** — some publisher PDFs embed figures as vector art; enable `document.ocr: true` fallback

## Roadmap

- Multi-paper comparison & paper-to-paper diffing
- Semantic search across a personal paper library
- Citation graph exploration
- Automatic literature-review drafts
- Zotero integration
- Knowledge graph construction
- Optional cloud LLM backends (clearly flagged when data leaves the machine)
- Automatic presentation generation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
