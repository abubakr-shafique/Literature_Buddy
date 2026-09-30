from literature_buddy.document.chunker import split_long


def test_chunks_respect_sections_and_pages(paper):
    doc = paper.document
    for c in paper.store.chunks:
        if c.kind == "text":
            secs = {s.title for s in doc.sections if s.title == c.section}
            assert secs
    kinds = {c.kind for c in paper.store.chunks}
    assert {"abstract", "text", "figure", "table", "equation", "reference"} <= kinds


def test_visual_chunks_carry_labels_and_mentions(paper):
    fig = paper.store.by_label("Figure 1")[0]
    assert "Referred to in the text" in fig.text
    methods = [c for c in paper.store.chunks if c.section == "Methods"][0]
    assert "Equation 1" in methods.refs


def test_split_long_overlap():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(60))
    parts = split_long(text, 60)
    assert len(parts) > 2 and all(len(p.split()) <= 80 for p in parts)
    assert parts[0].split(".")[-2].strip() in parts[1]  # 1-sentence overlap
