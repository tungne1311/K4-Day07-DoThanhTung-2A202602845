from typing import Callable

from .store import EmbeddingStore

NO_CONTEXT_MESSAGE = (
    "Không tìm thấy tài liệu liên quan trong cơ sở tri thức để trả lời câu hỏi này."
)

PROMPT_TEMPLATE = """Bạn là trợ lý tra cứu quy định. Chỉ được dùng NGỮ CẢNH bên dưới để trả lời.

Quy tắc bắt buộc:
- Chỉ dùng thông tin có trong ngữ cảnh. Không suy đoán, không bổ sung kiến thức ngoài.
- Trích dẫn số nguồn dạng [1], [2] ngay sau thông tin lấy từ nguồn đó.
- Nếu ngữ cảnh không đủ để trả lời, nói rõ là không tìm thấy thông tin.
- Giữ nguyên con số, mốc thời gian và đơn vị đúng như trong ngữ cảnh.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)

        # Store rỗng hoặc không truy xuất được gì: trả thông báo, không gọi LLM vô ích.
        if not results:
            return NO_CONTEXT_MESSAGE

        blocks = []
        for position, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            # Đánh số [1] [2] [3] kèm nguồn để câu trả lời truy vết được về đúng
            # chunk và đúng file — tiêu chí Source Traceability trong docs/EVALUATION.md.
            source = metadata.get("source_url") or metadata.get("doc_id") or result.get("id", "")
            blocks.append(f"[{position}] (nguồn: {source})\n{result['content']}")

        prompt = PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)
        return self.llm_fn(prompt)
