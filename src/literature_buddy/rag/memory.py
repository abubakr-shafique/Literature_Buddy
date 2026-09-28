"""Short-term conversation memory (spec §10): windowed turns with a summarized
prefix so follow-ups ('How large was it?') resolve, without flooding context."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    max_turns: int = 6
    summary_threshold: int = 4
    turns: list[dict] = field(default_factory=list)  # {"role", "content"}
    summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def messages(self) -> list[dict]:
        window = self.turns[-self.max_turns * 2:]
        msgs = []
        if self.summary:
            msgs.append({"role": "system",
                         "content": f"Earlier discussion summary:\n{self.summary}"})
        return msgs + window

    def maybe_summarize(self, llm_summarizer=None) -> None:
        """Keep the context lean: collapse oldest overflow turns into the summary."""
        full = len(self.turns) > self.max_turns * 2
        target_len = self.summary_threshold
        if full and llm_summarizer is not None:
            old = self.turns[: len(self.turns) - self.max_turns * 2]
            text = "\n".join(f"{t['role']}: {t['content']}" for t in old)
            self.summary = llm_summarizer(text)[:800]
            self.turns = self.turns[len(old):]
        elif len(self.summary) > target_len * 200 and llm_summarizer is None:
            self.summary = self.summary[: target_len * 200]