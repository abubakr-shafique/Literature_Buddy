from literature_buddy.document.chunker import chunk_document
from literature_buddy.document.models import (Figure, ParsedDocument, Section,
                                              SectionKind, Table)


def _doc():
    d = ParsedDocument(source_path="/tmp/x.pdf", num_pages=3)
    d.page_texts = ["Intro text. " * 60, "Methods text. " * 60, "References\n[1] Smith et al."]
    d.sections = [Section(SectionKind.INTRODUCTION, "1. Introduction", 1, 1),
                  Section(SectionKind.METHODS, "2. Methods", 2, 2),
                  Section(SectionKind.REFERENCES, "References", 3, 3)]
    d.figures = [Figure("Figure 1", "Overview of method.", 2, "/tmp/fig1.png")]
    d.tables = [Table("Table 1", "Results.", "a | b\n1 | 2", 2)]
    return d


def test_chunker_respects_metadata():
    chunks = chunk_document(_doc(), max_chunk_tokens=100)
    assert all(c.page in (1, 2) for c in chunks)          # refs page excluded
    fig = [c for c in chunks if c.chunk_type == "figure_caption"]
    assert fig and fig[0].figure_label == "Figure 1"
    assert any(c.section == SectionKind.METHODS.value for c in chunks)