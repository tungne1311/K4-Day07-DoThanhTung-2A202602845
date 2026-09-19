from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Lookbehind: cắt ở khoảng trắng ĐỨNG SAU dấu câu, giữ lại dấu câu.
        # Dùng [.!?]\s+ sẽ nuốt mất dấu câu và mọi chunk thành câu cụt.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        step = self.max_sentences_per_chunk
        return [" ".join(sentences[i : i + step]) for i in range(0, len(sentences), step)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: không có gì để cắt.
        if not current_text or not current_text.strip():
            return []
        # Base case 2: đã đủ nhỏ, không cần cắt tiếp.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        separator = remaining_separators[0] if remaining_separators else ""
        rest = remaining_separators[1:]

        # Base case 3: hết separator, hoặc separator là chuỗi rỗng -> cắt cứng.
        # Không thể dùng split("") vì Python ném ValueError: empty separator.
        if not separator:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        chunks: list[str] = []
        buffer = ""
        for piece in current_text.split(separator):
            if not piece:
                continue
            # Mảnh vẫn quá dài -> đệ quy xuống separator nhỏ hơn.
            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split(piece, rest))
                continue
            # Gom lên: nối các mảnh nhỏ liền kề cho tới sát chunk_size.
            candidate = piece if not buffer else buffer + separator + piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = piece

        if buffer:
            chunks.append(buffer)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            comparison[name] = {
                "count": count,
                # Chặn chia 0 khi text rỗng.
                "avg_length": (sum(len(c) for c in chunks) / count) if count else 0.0,
                "chunks": chunks,
            }
        return comparison
