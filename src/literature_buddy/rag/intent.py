"""Rule-based question understanding (fast, deterministic, no LLM call)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..document.schema import find_mentions

_SUMMARY = re.compile(r"\b(summari[sz]e|summary|what is (this|the) paper about|main (contribution|idea|finding)s?|"
                      r"overview|tl;?dr|key (findings?|takeaways?)|in a nutshell)\b", re.I)
_CRITICAL = re.compile(r"\b(limitation|weakness|assumption|bias|biases|missing|confound|critique|critic|flaw|"
                       r"generali[sz]|threat|shortcoming|caveat|improve|robust)\w*", re.I)
_METHODS = re.compile(r"\b(dataset|data set|cohort|train|trained|training|preprocess|baseline|implementation|"
                      r"hyper-?parameter|inclusion|exclusion|criteria|protocol|architecture|optimi[sz]er|method)\w*", re.I)
_RESULTS = re.compile(r"\b(result|perform|accuracy|outperform|best|improv|score|metric|auc|dice|f1)\w*", re.I)
_VISUAL = re.compile(r"\b(figure|fig\.|plot|graph|chart|image|diagram|trend|curve|panel|heatmap|micrograph)\b", re.I)
_REFS = re.compile(r"\b(reference|references|cite|cited|citation|bibliograph)\w*", re.I)


@dataclass
class Intent:
    kind: str = "general"  # summary | visual | critical | general
    prefer_sections: set[str] = field(default_factory=set)
    wants_references: bool = False
    labels: list[str] = field(default_factory=list)  # explicit "Figure 3" style mentions


def classify(question: str) -> Intent:
    labels = find_mentions(question)
    it = Intent(labels=labels, wants_references=bool(_REFS.search(question)))
    if labels or _VISUAL.search(question):
        it.kind = "visual"
    if _SUMMARY.search(question) and not labels:
        it.kind = "summary"
        it.prefer_sections = {"abstract", "introduction", "conclusion", "discussion"}
    elif _CRITICAL.search(question):
        it.kind = "critical" if it.kind != "visual" else "visual"
        it.prefer_sections |= {"discussion", "limitations", "conclusion", "methods"}
    if _METHODS.search(question):
        it.prefer_sections |= {"methods"}
    if _RESULTS.search(question):
        it.prefer_sections |= {"results"}
    return it
