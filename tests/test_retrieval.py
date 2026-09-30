from literature_buddy.rag.memory import ConversationMemory
from literature_buddy.retrieval.bm25 import BM25, tokenize
from literature_buddy.retrieval.embeddings import HashingEmbedder
from literature_buddy.retrieval.multimodal import select_visuals
from literature_buddy.retrieval.retriever import HybridRetriever


def _retriever(paper, cfg, reranker=None):
    return HybridRetriever(paper.store, HashingEmbedder(), reranker, cfg.retrieval, paper.document.title)


def test_bm25_ranks_exact_terms():
    docs = [tokenize(t) for t in ["cats and dogs", "TCGA cohort samples", "unrelated text here"]]
    s = BM25(docs).scores(tokenize("TCGA samples"))
    assert s.argmax() == 1


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed_query("trained on TCGA"), e.embed_query("trained on TCGA")
    assert (a == b).all() and abs((a @ a) - 1) < 1e-4


def test_dataset_question_hits_methods(paper, cfg):
    hits = _retriever(paper, cfg).retrieve("How many samples were used for training?", top_k=3)
    assert hits[0].chunk.section == "Methods"


def test_limitation_question_hits_discussion(paper, cfg):
    hits = _retriever(paper, cfg).retrieve("limitation of the study scanners", top_k=3)
    assert hits[0].chunk.section == "Discussion"


def test_references_excluded_by_default(paper, cfg):
    hits = _retriever(paper, cfg).retrieve("deep residual learning", exclude_kinds={"reference"})
    assert all(h.chunk.kind != "reference" for h in hits)


def test_reranker_can_reorder(paper, cfg):
    class Rev:
        def score(self, q, passages):
            return [float(i) for i in range(len(passages))]  # prefers the last candidate

    base = _retriever(paper, cfg).retrieve("scanner", top_k=4)
    rer = _retriever(paper, cfg, Rev()).retrieve("scanner", top_k=4)
    assert rer[0].rerank_score is not None and rer[0].chunk.chunk_id != base[0].chunk.chunk_id


def test_visual_selection_priorities(paper, cfg):
    hits = _retriever(paper, cfg).retrieve("what does the contrastive result show", top_k=4)
    v = select_visuals(paper.document, "What does Table 1 show?", hits, 3)
    assert v[0].item.label == "Table 1" and v[0].reason == "asked"
    labels = [x.item.label for x in select_visuals(paper.document, "hello", hits, 3)]
    assert len(labels) == len(set(labels))


def test_memory_window_and_summary():
    m = ConversationMemory(budget_tokens=60)
    for i in range(8):
        m.add("user", f"question {i} " + "word " * 30)
        m.add("assistant", f"answer {i} " + "word " * 30)
    m.compress(lambda prev, text: "SUMMARY")
    assert m.summary == "SUMMARY" and len(m.turns) < 16
    assert m.messages()[0].content.startswith("Summary of earlier conversation")
