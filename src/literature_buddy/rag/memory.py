"""Conversation memory for chat."""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Message:
    """A chat message."""
    role: str  # "user" or "assistant"
    content: str


class ConversationMemory:
    """Stores conversation history."""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: List[Message] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content))
        
        # Trim if too long
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_history(self) -> str:
        """Get formatted conversation history."""
        if not self.messages:
            return ""
        
        lines = []
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        
        return "\n\n".join(lines)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []