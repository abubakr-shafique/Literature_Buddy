"""Short-term conversation memory: recent turns verbatim + rolling summary of older ones."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..models.base import ChatMessage


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    budget_tokens: int = 1500
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role, content))

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""

    @property
    def has_turns(self) -> bool:
        return bool(self.turns)

    def _fit(self) -> list[Turn]:
        kept: list[Turn] = []
        used = est_tokens(self.summary) if self.summary else 0
        for t in reversed(self.turns):
            cost = est_tokens(t.content)
            if kept and used + cost > self.budget_tokens:
                break
            kept.append(t)
            used += cost
        return list(reversed(kept))

    def messages(self) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        if self.summary:
            out.append(ChatMessage("system", f"Summary of earlier conversation: {self.summary}"))
        out += [ChatMessage(t.role, t.content) for t in self._fit()]
        return out

    def compress(self, summarizer: Callable[[str, str], str] | None) -> None:
        """Fold turns that no longer fit into the summary (summarizer(prev_summary, text) -> str)."""
        fit = self._fit()
        dropped = self.turns[: len(self.turns) - len(fit)]
        if not dropped:
            return
        text = "\n".join(f"{t.role}: {t.content[:600]}" for t in dropped)
        if summarizer is not None:
            try:
                self.summary = summarizer(self.summary, text).strip()
            except Exception:  # noqa: BLE001 - fall back to truncation below
                self.summary = (self.summary + " " + text)[-600:]
        else:
            self.summary = (self.summary + " " + text)[-600:]
        self.turns = fit

    def as_text(self, max_turns: int = 4) -> str:
        return "\n".join(f"{t.role}: {t.content[:400]}" for t in self.turns[-max_turns:])
