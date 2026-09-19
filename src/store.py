from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        # Chỉ dùng in-memory. Không test nào cần ChromaDB, requirements.txt không
        # cài nó, và code khởi tạo gốc gán self._use_chroma = True TRƯỚC khi client
        # được tạo — máy nào tình cờ có chromadb sẽ rẽ vào nhánh chưa cài đặt.
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Copy metadata thay vì dùng thẳng dict của người gọi: nếu không, sửa
        # record sau này sẽ sửa luôn Document gốc ở ngoài store.
        metadata = dict(doc.metadata or {})
        # delete_document() lọc theo metadata['doc_id'], nên khoá này phải luôn có.
        # Ở bench.py, Document.id là "ten-file#3" còn doc_id trỏ về "ten-file".
        metadata.setdefault("doc_id", doc.id)

        record = {
            "id": doc.id,
            "index": self._next_index,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        # Vector đã được chuẩn hoá (||v|| = 1) nên dot product bằng đúng cosine.
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                # Bỏ "embedding" khỏi kết quả: vector làm bẩn output khi in ra terminal.
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # Không tự chunk: 1 Document = 1 record. Chunking do tầng ngoài (bench.py) làm.
        for doc in docs or []:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            candidates = self._store
        else:
            # Lọc TRƯỚC rồi mới search. Nếu search trước rồi mới bỏ cái không khớp,
            # k slot có thể bị chiếm hết bởi tài liệu sai và trả về 0 kết quả dù
            # store vẫn còn tài liệu hợp lệ.
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        remaining = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        removed = len(remaining) != len(self._store)
        self._store = remaining
        return removed
