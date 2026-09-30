# Development

```bash
pip install -r requirements-dev.txt && pip install -e . --no-deps
export QT_QPA_PLATFORM=offscreen
pytest                                   # 49 tests, ~10 s, no GPU, no downloads
ruff check src tests scripts
python scripts/make_sample_pdf.py out.pdf
```

## Test strategy

| File | Covers |
|---|---|
| `test_parser.py` | title/authors/abstract, sections, figure box above caption, ruled table → Markdown, equation, references, column ordering, "Figure 1 shows…" not mistaken for a caption |
| `test_chunker.py`, `test_retrieval.py` | section/page-safe chunks, visual chunks with mentions, BM25, hybrid ranking, reranker hook, memory |
| `test_rag.py` | streaming, citation stripping, quote verification, highlights as context, VLM call + caching, no-evidence path, follow-up rewrite, VLM text never counted as paper text |
| `test_loader_models.py` | URL rewriting, SSRF guards, PDF validation, Ollama streaming via mock transport |
| `test_gui.py` | the full UI headlessly against `tests/fake_ollama.py`: open → highlight → ask → cite → reopen from cache → error paths |

## Adding a backend

1. Subclass `LLMBackend` (`stream()` required; add `VLMBackend` for images) in `src/literature_buddy/models/`.
2. Add the provider literal in `config/settings.py` and a branch in `models/model_manager.create_backend`.
3. Reuse `ThinkStripper` to remove reasoning blocks and honour `stop_event` in the stream loop.

## Tuning parsing

Run `literature-buddy index paper.pdf` and inspect `<data_dir>/papers/<id>/document.json` and `figures/`. Most heuristics are named constants/regexes at the top of `document/metadata.py` (section kinds, headings) and `document/figure_extractor.py` (caption regex, cluster gap, distances).

## Tuning prompts

All prompt text is in `rag/prompts.py`. After changes, add a case to `tests/test_rag.py` for the behaviour you want (the fake LLM lets you assert on the exact prompt).
