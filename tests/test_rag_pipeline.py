from literature_buddy.document.models import Chunk
from literature_buddy.rag.memory import ConversationMemory
from literature_buddy.rag.pipeline import RAGPipeline


class FakeLLM:
    def generate(self, messages, max_tokens=10, temperature=0.0):
        return "[SOURCE: Paper] They used TCGA [p.3, Methods]."
    def unload(self): pass


class FakeManager:
    def llm(self): return FakeLLM()


class FakeRetriever:
    class _Hit:
        def __init__(self):
            self.chunk = Chunk(text="We used TCGA.", page=3,
                               section_title="2. Methods")
            self.score = 0.9
        is_visual = False
    class _Res:
        chunks = [None]
    def retrieve(self, slug, q):
        r = self._Res(); return_less = self._Hit(); r.chunks = [return_less]
        return r


class FakeDoc:
    slug = "abc123"


def test_pipeline_returns_grounded_answer_with_citations():
    pipe = RAGPipeline(FakeDoc(), FakeRetriever(), FakeManager(),
                       ConversationMemory(), context_tokens=1_000)
    text, cits = pipe.answer("What dataset did they use?")
    assert "[SOURCE: Paper]" in text
    assert cits and cits[0].page == 3
    assert pipe.memory.turns[-1]["role"] == "assistant"