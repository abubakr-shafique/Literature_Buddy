#!/usr/bin/env python3
"""Generate a small synthetic 'scientific paper' PDF (PyMuPDF only) for demos and tests.

    python scripts/make_sample_pdf.py sample_paper.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

W, H = 612, 792
L, R = 66, 546


def _para(page: pymupdf.Page, y0: float, y1: float, text: str, size: float = 10.0) -> None:
    page.insert_textbox(pymupdf.Rect(L, y0, R, y1), text, fontsize=size, fontname="helv")


def _heading(page: pymupdf.Page, y: float, text: str) -> None:
    page.insert_text((L, y), text, fontsize=12, fontname="hebo")


def make_sample_pdf(path: str | Path) -> Path:
    path = Path(path)
    doc = pymupdf.open()

    # ---- page 1 -----------------------------------------------------------------------
    p = doc.new_page(width=W, height=H)
    p.insert_textbox(pymupdf.Rect(L, 50, R, 110),
                     "Domain-Adaptive Segmentation of Histopathology Slides with Contrastive Pretraining",
                     fontsize=18, fontname="hebo")
    p.insert_text((L, 128), "Alice Johnson, Bob Smith and Carla Gomez", fontsize=10.5, fontname="helv")
    p.insert_text((L, 146), "Department of Computer Science, Example University", fontsize=9, fontname="helv")
    _heading(p, 182, "Abstract")
    _para(p, 190, 290,
          "Segmentation models for histopathology often fail when the scanner or staining protocol "
          "changes. We propose a contrastive pretraining strategy that reduces this domain shift. "
          "Our method improves the mean Dice score by 6.2 points on unseen scanners compared with "
          "a supervised baseline, while requiring no labels from the target domain.")
    _heading(p, 312, "1. Introduction")
    _para(p, 320, 420,
          "Computational pathology relies on deep networks trained on whole-slide images. However, "
          "differences between scanners cause a substantial drop in accuracy at deployment time. "
          "Prior work has tried stain normalisation and adversarial alignment. In this paper we "
          "study contrastive pretraining as an alternative. Figure 1 summarises our main result and "
          "Table 1 describes the cohorts used in the experiments.")
    _heading(p, 442, "2. Methods")
    _para(p, 450, 545,
          "We trained the model using 12,450 samples from the TCGA cohort. Inclusion criteria were "
          "adult patients with a confirmed carcinoma diagnosis and at least one hematoxylin and eosin "
          "stained slide scanned at 40x magnification. Slides with pen marks or out-of-focus regions "
          "were excluded. The training objective is given in Equation 1.")
    p.insert_text((150, 566), "L = - sum_i log p(y_i | x_i)                                   (1)",
                  fontsize=10, fontname="helv")
    _para(p, 582, 680,
          "The encoder is a ResNet-50 pretrained for 200 epochs with a contrastive loss. Fine-tuning "
          "used the AdamW optimiser with a learning rate of 1e-4 and a batch size of 32. Baseline "
          "methods were a supervised ResNet-50 and a stain-normalised variant of the same network.")
    p.insert_text((W / 2 - 3, 765), "1", fontsize=9, fontname="helv")

    # ---- page 2 -----------------------------------------------------------------------
    p = doc.new_page(width=W, height=H)
    # vector bar chart (figure)
    x0, y0, x1, y1 = 120, 70, 480, 220
    p.draw_line((x0, y1), (x1, y1), width=1)
    p.draw_line((x0, y0), (x0, y1), width=1)
    for i, (h, col) in enumerate([(70, (0.6, 0.6, 0.6)), (110, (0.2, 0.4, 0.8)), (60, (0.6, 0.6, 0.6)),
                                  (120, (0.2, 0.4, 0.8)), (55, (0.6, 0.6, 0.6)), (105, (0.2, 0.4, 0.8))]):
        bx = x0 + 20 + i * 55
        p.draw_rect(pymupdf.Rect(bx, y1 - h, bx + 40, y1), fill=col, color=None)
    for i, lab in enumerate(["Scanner A", "Scanner B", "Scanner C"]):
        p.insert_text((x0 + 22 + i * 110, y1 + 12), lab, fontsize=8, fontname="helv")
    p.insert_text((70, 150), "Dice", fontsize=8, fontname="helv")
    _para(p, 245, 290,
          "Figure 1. Segmentation accuracy on three unseen scanners. Grey bars show the supervised "
          "baseline and blue bars show our contrastive pretraining; higher Dice is better.", size=9)
    _heading(p, 312, "3. Results")
    _para(p, 320, 405,
          "As shown in Figure 1, contrastive pretraining outperforms the supervised baseline on all "
          "three scanners, with the largest gain on Scanner B. Table 1 lists the number of slides "
          "and patients in each cohort. Performance on the held-out scanner C improved from 0.55 "
          "to 0.66 mean Dice.")
    _para(p, 418, 436, "Table 1. Dataset statistics for the three cohorts.", size=9)
    rows = [["Cohort", "Slides", "Patients", "Scanner"], ["TCGA", "12450", "4100", "Aperio"],
            ["Hospital B", "830", "310", "Hamamatsu"], ["Hospital C", "610", "205", "Philips"]]
    tx, ty, cw, rh = 100, 442, 110, 20
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = pymupdf.Rect(tx + c * cw, ty + r * rh, tx + (c + 1) * cw, ty + (r + 1) * rh)
            p.draw_rect(cell, color=(0, 0, 0), width=0.8)
            p.insert_text((cell.x0 + 6, cell.y0 + 14), val, fontsize=9, fontname="hebo" if r == 0 else "helv")
    _heading(p, 560, "4. Discussion")
    _para(p, 568, 660,
          "Our results suggest that contrastive pretraining learns features that are robust to scanner "
          "differences. A limitation of this study is that only three scanners were evaluated, so the "
          "conclusions may not generalise to other vendors. We did not evaluate stain variation "
          "independently of the scanner.")
    p.insert_text((W / 2 - 3, 765), "2", fontsize=9, fontname="helv")

    # ---- page 3 -----------------------------------------------------------------------
    p = doc.new_page(width=W, height=H)
    _heading(p, 70, "5. Conclusion")
    _para(p, 78, 140,
          "We presented a contrastive pretraining approach that improves robustness of histopathology "
          "segmentation to scanner shift. Future work will study additional tissue types.")
    _heading(p, 170, "References")
    refs = [
        "[1] K. He, X. Zhang, S. Ren and J. Sun. Deep residual learning for image recognition. CVPR, 2016.",
        "[2] T. Chen, S. Kornblith, M. Norouzi and G. Hinton. A simple framework for contrastive learning "
        "of visual representations. ICML, 2020.",
        "[3] M. Macenko et al. A method for normalizing histology slides for quantitative analysis. ISBI, 2009.",
    ]
    y = 180
    for r in refs:
        p.insert_textbox(pymupdf.Rect(L, y, R, y + 30), r, fontsize=9, fontname="helv")
        y += 32
    p.insert_text((W / 2 - 3, 765), "3", fontsize=9, fontname="helv")

    doc.save(path)
    doc.close()
    return path


def make_two_column_pdf(path: str | Path) -> Path:
    """One page, two text columns: reading order must be left column, then right column."""
    path = Path(path)
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    p.insert_textbox(pymupdf.Rect(L, 50, R, 90), "Two Column Layout Test", fontsize=18, fontname="hebo")
    left = "LEFTCOLUMN " + "alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 12
    right = "RIGHTCOLUMN " + "one two three four five six seven eight nine ten. " * 12
    p.insert_textbox(pymupdf.Rect(L, 120, 296, 400), left, fontsize=10, fontname="helv")
    p.insert_textbox(pymupdf.Rect(316, 120, R, 400), right, fontsize=10, fontname="helv")
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = make_sample_pdf(sys.argv[1] if len(sys.argv) > 1 else "sample_paper.pdf")
    print(f"Wrote {out}")
