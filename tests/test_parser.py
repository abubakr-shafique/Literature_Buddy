from literature_buddy.document.layout import extract_blocks, reading_order
from literature_buddy.document.schema import find_mentions


def test_metadata(paper):
    d = paper.document
    assert d.title.startswith("Domain-Adaptive Segmentation")
    assert d.authors == ["Alice Johnson", "Bob Smith", "Carla Gomez"]
    assert "contrastive pretraining" in d.abstract
    assert d.page_count == 3


def test_sections(paper):
    kinds = [s.kind for s in paper.document.sections]
    for k in ("abstract", "introduction", "methods", "results", "discussion", "conclusion", "references"):
        assert k in kinds
    methods = next(s for s in paper.document.sections if s.kind == "methods")
    assert methods.page_start == 0


def test_figure_and_caption(paper):
    f = paper.document.visual("Figure 1")
    assert f and f.page == 1 and f.bbox is not None
    assert f.caption.startswith("Figure 1.") and "Dice" in f.caption
    assert (paper.directory / f.image_path).exists()
    assert f.bbox[3] < f.caption_bbox[1] + 8  # figure sits above its caption


def test_table(paper):
    t = paper.document.visual("Table 1")
    assert t and (t.n_rows, t.n_cols) == (4, 4) and not t.approximate
    assert "| TCGA | 12450 | 4100 | Aperio |" in t.markdown


def test_equation_and_references(paper):
    d = paper.document
    assert d.visual("Equation 1") is not None
    assert len(d.references) == 3 and d.references[0].text.startswith("K. He")


def test_running_text_is_not_a_caption(paper):
    # "Figure 1 summarises ..." / "As shown in Figure 1" must stay body text
    assert len(paper.document.figures) == 1
    assert any("As shown in Figure 1" in p.text for p in paper.document.paragraphs)


def test_page_numbers_removed(paper):
    assert all(p.text.strip() not in {"1", "2", "3"} for p in paper.document.paragraphs)


def test_two_column_reading_order(two_col_pdf):
    import pymupdf

    with pymupdf.open(two_col_pdf) as pdf:
        page = pdf[0]
        blocks = [b for b in extract_blocks(page, 0) if b.size < 12]
        ordered = reading_order(blocks, page.rect.width)
    text = " ".join(b.text for b in ordered)
    assert text.index("LEFTCOLUMN") < text.index("RIGHTCOLUMN")
    assert text.rindex("alpha") < text.index("RIGHTCOLUMN")


def test_mentions():
    assert find_mentions("see Fig. 3A, Figures 2 and Table 4; Eq. (5)") == ["Figure 3", "Figure 2", "Table 4", "Equation 5"]
