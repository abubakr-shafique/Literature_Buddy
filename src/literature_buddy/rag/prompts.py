"""Prompt templates for RAG."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from academic literature.

Guidelines:
- Answer based on the context provided
- Cite specific parts when possible
- If the context doesn't contain the answer, say so
- Be concise but thorough
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer:"""

CHAT_PROMPT_TEMPLATE = """<|system|>
{system_message}

<|user|>
{user_message}

<|assistant|>
"""