
from literature_buddy.models.base import VLMBackend
from literature_buddy.rag.context_builder import build_context
from literature_buddy.rag.pipeline import RagPipeline
from literature_buddy.retrieval.embeddings import HashingEmbedder
from literature_buddy.retrieval.retriever import HybridRetriever


class FakeLLM(VLMBackend):
    def __init__(self, reply="The model used 12,450 samples [S1]."):
        self.reply, self.calls, self.images = reply, [], []

    def stream(self, messages, **kw):
        self.calls.append(list(messages))
        self.images += [i for m in messages for i in m.images]
        yield from (self.reply[i:i + 8] for i in range(0, len(self.reply), 8))


class FakeManager:
    def __init__(self, llm, vlm=None):
        self._l, self._v = llm, vlm

    def llm(self):
        return self._l

    def vlm(self):
        return self._v


def _pipe(paper, cfg, llm, vlm=None):
    r = HybridRetriever(paper.store, HashingEmbedder(), None, cfg.retrieval, paper.document.title)
    return RagPipeline(cfg, paper, r, FakeManager(llm, vlm))


def _run(pipe, q, **kw):
    events = list(pipe.ask(q, **kw))
    assert events[-1].kind == "done"
    return events[-1].result, events


def test_answer_streams_and_cites(paper, cfg):
    res, events = _run(_pipe(paper, cfg, FakeLLM()), "How many samples were used for training?")
    assert "".join(e.text for e in events if e.kind == "token") == "The model used 12,450 samples [S1]."
    assert res.used == ["S1"] and res.display_sources[0].sid == "S1"


def test_fabricated_citation_removed(paper, cfg):
    res, _ = _run(_pipe(paper, cfg, FakeLLM("Yes [S1] and also [S42].")), "samples for training?")
    assert "[S42]" not in res.text and any("non-existent" in w for w in res.warnings)


def test_unverified_quote_flagged(paper, cfg):
    llm = FakeLLM('The paper says "we trained on one billion unicorn images every day" [S1].')
    res, _ = _run(_pipe(paper, cfg, llm), "samples for training?")
    assert any("could not be verified" in w for w in res.warnings)


def test_highlights_become_sources(paper, cfg):
    hl = [{"page": 0, "text": "SPECIAL HIGHLIGHT TEXT about inclusion", "use_in_chat": True},
          {"page": 1, "text": "IGNORED", "use_in_chat": False}]
    llm = FakeLLM("ok [H1]")
    res, _ = _run(_pipe(paper, cfg, llm), "inclusion criteria?", highlights=hl)
    prompt = llm.calls[-1][-1].content
    assert "[H1]" in prompt and "SPECIAL HIGHLIGHT" in prompt and "IGNORED" not in prompt
    assert res.used == ["H1"]


def test_figure_question_sends_image_to_vlm(paper, cfg):
    llm = FakeLLM("Grey vs blue bars [S1]")
    vlm = FakeLLM("The chart shows six bars; blue bars are higher.")
    pipe = _pipe(paper, cfg, llm, vlm)
    res, events = _run(pipe, "What does Figure 1 show?")
    assert len(vlm.images) == 1 and vlm.images[0][:4] == b"\x89PNG"
    assert "MODEL-GENERATED VISUAL ANALYSIS" in llm.calls[-1][-1].content
    assert any(e.kind == "status" and "Figure 1" in e.text for e in events)
    # cached: a second identical question does not call the VLM again
    _run(pipe, "What does Figure 1 show?")
    assert len(vlm.images) == 1


def test_no_vlm_falls_back_to_caption_with_warning(paper, cfg):
    res, _ = _run(_pipe(paper, cfg, FakeLLM("Caption says so [S1]"), None), "What does Figure 1 show?")
    assert any("Vision model unavailable" in w for w in res.warnings)


def test_no_evidence_message(paper, cfg, monkeypatch):
    pipe = _pipe(paper, cfg, FakeLLM())
    monkeypatch.setattr(pipe.retriever, "retrieve", lambda *a, **k: [])
    res, _ = _run(pipe, "hello there")
    assert res.text.startswith("I could not find evidence")


def test_followup_is_rewritten(paper, cfg):
    llm = FakeLLM("They used TCGA [S1]")
    pipe = _pipe(paper, cfg, llm)
    _run(pipe, "What dataset did they use?")
    llm.reply = "TCGA training samples"
    _, _ = _run(pipe, "How large was it?")
    assert any("Standalone query" in c[0].content for c in llm.calls)


def test_context_budget(paper, cfg):
    hits = HybridRetriever(paper.store, HashingEmbedder(), None, cfg.retrieval).retrieve("model", top_k=8)
    ctx, srcs = build_context(paper.document, hits, [], {}, [], max_tokens=60)
    assert 1 <= len(srcs) < len(hits)


def test_vlm_note_is_not_paper_text(paper, cfg):
    """Vision output reaches the LLM but never counts as paper text (excerpts, quote verification)."""
    llm = FakeLLM('Bars: "the blue bars are higher than the grey ones every time" [S1]')
    vlm = FakeLLM("the blue bars are higher than the grey ones every time")
    res, _ = _run(_pipe(paper, cfg, llm, vlm), "What does Figure 1 show?")
    fig = next(s for s in res.sources if s.label == "Figure 1")
    assert fig.note and "MODEL-GENERATED" not in fig.excerpt and fig.note not in fig.text
    assert any("could not be verified" in w for w in res.warnings)
