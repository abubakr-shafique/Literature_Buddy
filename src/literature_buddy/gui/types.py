from __future__ import annotations

from dataclasses import dataclass, field

Rect = tuple[float, float, float, float]


@dataclass
class Highlight:
    hid: str
    page: int  # 0-based
    rects: list[Rect]
    text: str
    use_in_chat: bool = True

    def as_context(self) -> dict:
        return {"page": self.page, "text": self.text, "use_in_chat": self.use_in_chat}


@dataclass
class ViewState:
    scale: float = 1.3
    search_hits: dict[int, list[Rect]] = field(default_factory=dict)
    evidence: dict[int, list[Rect]] = field(default_factory=dict)
