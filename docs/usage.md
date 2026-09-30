# User guide

## 1. Start

```bash
literature-buddy                      # opens the window with a drop zone
literature-buddy paper.pdf            # or open a paper directly
literature-buddy https://arxiv.org/abs/2401.01234
```

The bottom-right corner shows the model status: **● green** = ready, **○ red** = the backend is not reachable (the message tells you what to run, e.g. `ollama serve` or `ollama pull …`).

## 2. Opening papers

| How | Notes |
|---|---|
| Click the grey panel / `Ctrl+O` | Local PDF |
| Drag & drop | PDF file, or a link dragged from a browser |
| `Ctrl+L` | arXiv, bioRxiv/medRxiv, PubMed Central, or any page with a `citation_pdf_url` tag |
| File → Recent papers | Reopens instantly from the cache, restoring chat and highlights |

Publishers that block automated downloads will show an error: download the PDF in your browser and open it from disk.

## 3. Reading tools

`Ctrl+wheel` zoom · `Ctrl+= / Ctrl+- / Ctrl+0` zoom / fit width · page box + ◀ ▶ · `Ctrl+F` search (Enter = next).

## 4. Highlights as chat context

* Drag across text → highlighted (yellow) and listed under **Highlights** with a ☑.
* ☑ = sent to the model with your next questions (top priority, cited as `[H1]`…). Untick to keep the mark but stop using it (turns grey).
* **Clear selected** (select rows first) or **Clear all**; or right-click a highlight in the PDF.
* Double-click a row to jump to it.

Good uses: highlight a confusing paragraph and ask *"Explain this in simpler terms"*; highlight two passages and ask *"Do these contradict each other?"*.

## 5. Asking good questions

| You want | Ask like |
|---|---|
| Overview | "Summarize this paper" · "What is the main contribution?" |
| Methods | "What dataset, inclusion criteria and baselines were used?" |
| A figure | "What does Figure 3 show?" · "Which group performs best in Figure 2?" |
| A table | "What is the difference between rows 2 and 3 in Table 1?" |
| Critique | "What are the limitations and hidden assumptions?" |
| Follow-ups | "How large was it?" (pronouns are resolved from the conversation) |

Reading the answer: green **Paper states** = written in the paper (with citations); orange **Inference** = reasoning or figure reading (fallible); purple **General scientific context** = not from the paper. Warnings (⚠) mean a citation was removed, a quote could not be verified, or the vision model was unavailable.

## 6. Verifying

Click `[S1]` or a source card: the viewer scrolls to the page and outlines the passage (text) or figure/table box. Figure sources include a thumbnail of what the vision model saw.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot reach Ollama" | Install from ollama.com, run `ollama serve`, then Models → Check backend |
| "Model … is not installed" | `ollama pull <name>` shown in the message |
| Red "Model: PyTorch/transformers are not installed" | `pip install -r requirements-transformers.txt` (Transformers profiles) or use an `-ollama` profile |
| First question is slow (Transformers) | The model loads on first use; later questions are fast |
| Out of memory | Use the smaller profile, lower `llm.context_length` / `retrieval.max_context_tokens`, or Models → Unload |
| "Neural embeddings unavailable, using lexical fallback" | Answers still work but retrieval is weaker; fix the embedding backend named in the message |
| No figures detected | Only figures with a "Figure N"/"Fig. N" caption are found; check `document.json` in the paper folder |
| Scanned PDF, no text | Install Tesseract so OCR can run, or use a text-based PDF |
| Wrong section/figure boundaries | Heuristic limits: open an issue with the PDF (if shareable) |

Logs: `<data_dir>/logs/literature_buddy.log`.
