# Architecture

## Layers

```mermaid
flowchart TB
    subgraph GUI [gui/ · PySide6]
      MW[MainWindow] --- PV[PdfViewer] & CW[ChatWidget] & HP[HighlightsPanel] & SD[SettingsDialog]
    end
    subgraph SVC [services.py]
      SV[Services]
    end
    subgraph DOC [document/]
      LD[loader] --> PR[parser] --> CH[chunker] --> PRC[processor]
    end
    subgraph RET [retrieval/]
      EM[embeddings] & VS[(vector_store)] & HR[hybrid retriever] & RR[reranker] & MM[multimodal]
    end
    subgraph RAG [rag/]
      IN[intent] --> PL[pipeline] --> CB[context_builder]
      PL --> CI[citations] & ME[memory]
    end
    subgraph MOD [models/]
      MGR[ModelManager] --> OL[Ollama] & OA[OpenAI-compat] & TF[Transformers LLM/VLM] & LC[llama.cpp]
    end
    ST[(storage/ · cache + library.sqlite)]
    GUI --> SVC --> DOC & RAG & ST
    RAG --> RET & MOD
    DOC --> RET
```

Dependencies point downwards only; `gui/` is the only layer that imports Qt (except `gui/tasks.py`), so the whole pipeline is usable headlessly (`literature-buddy ask …`) and testable without a display.

## Ingestion

```mermaid
flowchart LR
    A[PDF] --> B[extract blocks<br/>font size · bold · bbox] --> C[drop headers/footers<br/>+ page numbers]
    C --> D[column-aware reading order]
    D --> E[captions Figure/Table N]
    E --> F[tables: find_tables + text fallback]
    E --> G[figures: graphics grid-clustering<br/>above caption]
    D --> H[headings → sections<br/>+ kind inheritance]
    F & G & H --> I[Document JSON]
    I --> J[chunks] --> K[embeddings] --> L[(index.sqlite)]
```

Key decisions:

| Decision | Why |
|---|---|
| Figure regions found by **clustering vector drawings and images on an occupancy grid**, then attaching the cluster above a caption | Works for raster and vector plots, multi-panel figures, and needs no layout model |
| Chunks **never cross a section or page boundary** | Every citation has one exact page; sections drive boosting (methods/results/discussion) |
| Figure/table chunks carry the sentences that *refer to* them | Lets a text question about "the result" retrieve the figure, and vice versa |
| **SQLite + numpy** instead of FAISS/Chroma/Qdrant | One paper = hundreds of chunks: exact search is instant, index is one portable file, zero extra deps. `SqliteVectorStore` is a narrow interface for a later LanceDB/Qdrant swap |
| **Own BM25** (30 lines) | No FTS5 availability issues, no dependency |
| One natively multimodal model for LLM+VLM (`vlm.same_as_llm`) | No VRAM swapping; `ModelManager` still unloads before switching if you configure two in-process models |
| Vision output kept in `Source.note`, never in `Source.text` | The model's reading of an image must not be able to "verify" a quotation |

## Answering a question

```mermaid
sequenceDiagram
    participant U as User
    participant P as RagPipeline
    participant R as HybridRetriever
    participant V as VLM
    participant L as LLM
    U->>P: question + ticked highlights
    P->>P: classify intent (summary/visual/critical/general)
    P->>L: rewrite follow-up → standalone query (only if needed)
    P->>R: retrieve(query, section preferences)
    R-->>P: hits (dense+BM25 → RRF → rerank)
    P->>P: select visuals (asked > retrieved > mentioned)
    opt asked figure, or retrieved visual question
        P->>V: image + caption + question
        V-->>P: MODEL-GENERATED note (cached)
    end
    P->>P: build numbered context [H*] [S*] within token budget
    P->>L: system rules + memory + sources + question
    L-->>P: streamed tokens
    P->>P: strip unknown [S#], flag unverifiable quotes
    P-->>U: answer + sources + warnings
```

### Answer contract (see `rag/prompts.py`)

`**Paper states:**` (cited facts) · `**Inference:**` (reasoned, incl. "(figure reading)") · `**General scientific context:**` (not from the paper) · fixed refusal sentence when evidence is missing.

## Data on disk

```
<data_dir>/papers/<sha256[:16]>/{source.pdf, document.json, index.sqlite, figures/*.png}
<data_dir>/library.sqlite   recents · chat history · highlights
<data_dir>/logs/
```
Caches are keyed by parser version + options and embedder id, so changing either re-builds only what is stale.

## Extending

* New LLM backend: subclass `LLMBackend` (and `VLMBackend` for images) in `models/`, register in `create_backend`.
* New parser: produce a `document.schema.Document`; everything downstream is parser-agnostic.
* New vector store: implement the methods used by `HybridRetriever` (`chunks`, `dense_scores`, `sparse_scores`, `by_label`).
