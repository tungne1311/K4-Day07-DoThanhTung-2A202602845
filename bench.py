"""Công cụ đo retrieval cho Lab 7 — K4-L3A, nhóm Top1Server, corpus thư viện HUIT.

Chạy (PowerShell — nhớ PYTHONIOENCODING để không hỏng tiếng Việt khi ghi ra file):

    python bench.py                       # fixed + LocalEmbedder (mặc định)
    python bench.py --strategy heading    # Nguyên
    python bench.py --strategy sentence   # Minh
    python bench.py --strategy recursive  # Đức Anh
    python bench.py --all                 # chạy cả 4, in bảng so sánh

    $env:PYTHONIOENCODING="utf-8"; python bench.py | Out-File -Encoding utf8 ket_qua_benchmark.txt

Backend mặc định là LocalEmbedder (sentence-transformers), KHÔNG cần API key.
Cài bằng: pip install -r requirements-local.txt

Đây là công cụ đo của nhóm, không phải bài tập được chấm bằng pytest.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.models import Document
from src.store import EmbeddingStore

CORPUS_DIR = Path("data/thu-vien-huit")
FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


class HeadingChunker:
    """Chiến lược chia nhỏ theo tiêu đề/mục — thiết kế riêng cho văn bản quy định.

    Lý do thiết kế: văn bản quy định được người soạn chia sẵn theo mục
    ("## 5. Quy định mượn/trả tài liệu"), mỗi mục đã là một đơn vị ngữ nghĩa
    trọn vẹn. Cắt theo ranh giới đó giữ được trọn điều khoản và bảng tra cứu,
    thay vì cắt ngang giữa một quy định như chunker theo ký tự.

    Ba bước:
      1. Cắt trước mỗi dòng heading, giữ lại chính dòng heading đó.
      2. Gộp section ngắn hơn min_chars vào section liền trước, tránh chunk rác.
      3. Section dài hơn max_chars thì hạ xuống RecursiveChunker, và tiêu đề được
         gắn lại vào từng mảnh con — nếu không, mảnh thứ hai trở đi mất ngữ cảnh
         "đây là mục nói về cái gì".
    """

    def __init__(self, max_chars: int = 800, min_chars: int = 200) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self._fallback = RecursiveChunker(chunk_size=max_chars)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Bước 1 — lookahead: cắt TRƯỚC dòng heading, không nuốt mất dòng đó.
        sections = [s.strip() for s in re.split(r"\n(?=#{1,6} )", text) if s.strip()]

        # Bước 2 — gộp section quá ngắn vào section trước đó.
        merged: list[str] = []
        for section in sections:
            if (
                merged
                and len(merged[-1]) < self.min_chars
                and len(merged[-1]) + 2 + len(section) <= self.max_chars
            ):
                merged[-1] = f"{merged[-1]}\n\n{section}"
            else:
                merged.append(section)

        # Bước 3 — section vẫn quá dài thì cắt nhỏ, gắn lại tiêu đề vào từng mảnh.
        chunks: list[str] = []
        for section in merged:
            if len(section) <= self.max_chars:
                chunks.append(section)
                continue

            lines = section.splitlines()
            heading = lines[0] if lines and lines[0].lstrip().startswith("#") else ""
            body = "\n".join(lines[1:]) if heading else section
            for piece in self._fallback.chunk(body):
                chunks.append(f"{heading}\n{piece}".strip() if heading else piece)
        return chunks


STRATEGIES = {
    "fixed": ("FixedSizeChunker(500, overlap 50)", "Đỗ Thanh Tùng", FixedSizeChunker(chunk_size=500, overlap=50)),
    "sentence": ("SentenceChunker(max 3 câu)", "Ninh Quang Minh", SentenceChunker(max_sentences_per_chunk=3)),
    "recursive": ("RecursiveChunker(500)", "Phạm Đức Anh", RecursiveChunker(chunk_size=500)),
    "heading": ("HeadingChunker(max 800, min 200)", "Trần Võ Hoàng Nguyên", HeadingChunker(max_chars=800, min_chars=200)),
}

# 5 benchmark query nhóm Top1Server đã thống nhất. Cả nhóm chạy chung bộ này.
# must_contain: các chuỗi đặc trưng phải CÙNG xuất hiện trong ngữ cảnh truy xuất
# được — chấm ở mức NỘI DUNG, không chỉ kiểm doc_id có trong top-3.
# Mọi chuỗi dưới đây đã grep xác nhận tồn tại nguyên văn trong corpus.
QUERIES = [
    {
        "dang": "tra số liệu + CÂU ĐÁNH BẪY (cần lọc metadata)",
        "query": "Được mượn tối đa bao nhiêu tài liệu và trong bao nhiêu ngày?",
        "gold_doc": "han-muc-muon-tai-lieu-sinh-vien",
        "gold_answer": "Sinh viên, học viên được mượn tối đa 3 tài liệu trong 10 ngày.",
        "must_contain": ["Sinh viên, học viên | 3 | 10"],
        "metadata_filter": {"audience": "student"},
    },
    {
        "dang": "hỏi điều kiện",
        "query": "Người sử dụng cần đáp ứng những điều kiện nào để được mượn tài liệu về nhà?",
        "gold_doc": "luu-hanh-tai-lieu",
        "gold_answer": "Hoàn thành bài kiểm tra hướng dẫn sử dụng thư viện đạt 25/35 câu, đăng ký thẻ thư viện và đóng tiền thế chân theo quy định.",
        "must_contain": ["25/35"],
        "metadata_filter": None,
    },
    {
        "dang": "hỏi quy trình + điều kiện hủy",
        "query": "Quy trình đăng ký và sử dụng phòng học nhóm gồm những bước nào, và đến trễ bao lâu thì kết quả đặt phòng bị hủy?",
        "gold_doc": "su-dung-phong-hoc-nhom",
        "gold_answer": "Đăng ký tại Quầy thông tin hoặc mục ĐẶT PHÒNG trực tuyến; nhận kết quả qua email; đến Quầy thông tin Lầu 3 hoặc Lầu 4 làm thủ tục. Đến trễ trên 15 phút thì kết quả bị hủy.",
        "must_contain": ["ĐẶT PHÒNG", "trễ trên 15 phút"],
        "metadata_filter": None,
    },
    {
        "dang": "tra số liệu ghép (3 con số)",
        "query": "Dịch vụ mượn liên thư viện cho phép mượn tối đa bao nhiêu tài liệu, trong bao lâu và phí trễ hạn là bao nhiêu?",
        "gold_doc": "muon-lien-thu-vien",
        "gold_answer": "Tối đa 2 tài liệu/lần, thời hạn 20 ngày từ ngày nhận, phí trễ hạn 5.000 đồng/tài liệu/ngày.",
        "must_contain": ["2 tài liệu/1 lần mượn", "20 ngày kể từ ngày nhận", "5.000 đồng/tài liệu/ngày"],
        "metadata_filter": None,
    },
    {
        "dang": "tra bảng giá + thời hạn",
        "query": "Lệ phí cấp mới, cấp lại, gia hạn thẻ thư viện là bao nhiêu và thời gian trả thẻ được quy định thế nào?",
        "gold_doc": "huong-dan-su-dung-thu-vien",
        "gold_answer": "Cấp mới 100.000 đồng, cấp lại 50.000 đồng, gia hạn 50.000 đồng/năm; thẻ mới trả sau bài kiểm tra 1 tuần hoặc theo lịch hẹn, thẻ cấp lại sau 7 ngày.",
        "must_contain": ["100.000 đ/thẻ", "07 ngày kể từ ngày đăng ký"],
        "metadata_filter": None,
    },
]


def parse_markdown(path: Path) -> tuple[dict, str]:
    """Tách front matter thành metadata, phần còn lại thành content."""
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    metadata = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key:
            metadata[key] = value.strip().strip('"')
    return metadata, text[match.end():].strip()


def build_store(strategy_name: str, embedding_fn) -> tuple[EmbeddingStore, int]:
    chunker = STRATEGIES[strategy_name][2]
    store = EmbeddingStore(collection_name="thu-vien-huit", embedding_fn=embedding_fn)

    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        metadata, body = parse_markdown(path)
        for i, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{path.stem}#{i}",
                    content=chunk,
                    # Front matter trải vào MỌI chunk, nếu không search_with_filter
                    # không có gì để lọc. doc_id trỏ về tên file gốc, không phải id chunk.
                    metadata={**metadata, "doc_id": path.stem, "source_file": path.name},
                )
            )

    store.add_documents(documents)
    return store, len(documents)


def score_query(spec: dict, results: list[dict]) -> tuple[int, str]:
    """Chấm 2 mức theo docs/SCORING.md: doc_id nằm trong top-3 CHƯA đủ,
    ngữ cảnh phải chứa đủ mọi chuỗi đặc trưng của gold answer."""
    doc_ids = [r["metadata"].get("doc_id") for r in results]
    context = "\n".join(r["content"] for r in results).lower()
    missing = [s for s in spec["must_contain"] if s.lower() not in context]

    if spec["gold_doc"] not in doc_ids:
        return 0, "0đ — gold doc không có trong top-3"
    if missing:
        return 0, f"0đ — đúng tài liệu nhưng ngữ cảnh THIẾU: {missing}"
    if doc_ids[0] == spec["gold_doc"]:
        return 2, "2đ — gold doc ở top-1 và ngữ cảnh chứa đủ đáp án"
    return 1, "1đ — gold doc ở top-2/3, ngữ cảnh chứa đủ đáp án"


def print_results(results: list[dict], indent: str = "    ") -> None:
    if not results:
        print(f"{indent}(không có kết quả)")
        return
    for rank, r in enumerate(results, start=1):
        preview = " ".join(r["content"].split())[:95]
        print(f"{indent}{rank}. score={r['score']:+.4f}  {r['metadata'].get('doc_id')}  [{r['id']}]")
        print(f"{indent}   {preview}...")


def make_embedding_fn(backend: str):
    """LocalEmbedder không cần API key. Mock chỉ dùng khi máy chưa cài được."""
    if backend == "local":
        from src.embeddings import LocalEmbedder

        return LocalEmbedder(), "LocalEmbedder (paraphrase-multilingual-MiniLM-L12-v2, 384 chiều)"
    from src.embeddings import MockEmbedder

    return MockEmbedder(), "MockEmbedder (băm MD5 — KHÔNG có ngữ nghĩa, số liệu là nhiễu)"


def run_strategy(name: str, embedding_fn, backend_label: str, top_k: int, verbose: bool = True) -> tuple[int, int]:
    label, owner, _ = STRATEGIES[name]
    store, chunk_count = build_store(name, embedding_fn)

    if verbose:
        print("=" * 78)
        print("BENCHMARK RETRIEVAL — K4-L3A, nhóm Top1Server")
        print(f"Corpus          : {CORPUS_DIR} ({len(list(CORPUS_DIR.glob('*.md')))} tài liệu)")
        print(f"Chiến lược chunk: {label}  —  {owner}")
        print(f"Số chunk đã nạp : {chunk_count}")
        print(f"Embedding       : {backend_label}")
        print("=" * 78)

    total = 0
    for i, spec in enumerate(QUERIES, start=1):
        results = store.search_with_filter(spec["query"], top_k=top_k, metadata_filter=spec["metadata_filter"])
        points, verdict = score_query(spec, results)
        total += points

        if verbose:
            print(f"\n[{i}] {spec['query']}")
            print(f"    dạng hỏi   : {spec['dang']}")
            print(f"    filter     : {spec['metadata_filter'] or '(không)'}")
            print(f"    gold doc   : {spec['gold_doc']}")
            print(f"    gold answer: {spec['gold_answer']}")
            print(f"    --- top-{top_k} ---")
            print_results(results)
            print(f"    => {verdict}")

    if verbose:
        print("\n" + "=" * 78)
        print(f"TỔNG ĐIỂM RETRIEVAL: {total}/10")
        print("=" * 78)

        # A/B bắt buộc: chạy câu đánh bẫy hai lần để chứng minh filter có tác dụng.
        ab = QUERIES[0]
        print(f"\nA/B METADATA FILTER — \"{ab['query']}\"")
        for lbl, mf in (("KHÔNG filter", None), ("CÓ filter audience=student", ab["metadata_filter"])):
            print(f"\n  {lbl}:")
            print_results(store.search_with_filter(ab["query"], top_k=top_k, metadata_filter=mf), indent="      ")

    return total, chunk_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval trên corpus thư viện HUIT.")
    parser.add_argument("--strategy", default="fixed", choices=sorted(STRATEGIES))
    parser.add_argument("--backend", default=os.getenv("EMBEDDING_PROVIDER", "local"), choices=["mock", "local"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Chạy cả 4 chiến lược, in bảng so sánh")
    args = parser.parse_args()

    try:
        embedding_fn, backend_label = make_embedding_fn(args.backend)
    except Exception as error:
        print(f"Không dùng được backend '{args.backend}': {error}", file=sys.stderr)
        print("Cài bằng: pip install -r requirements-local.txt — tạm quay về MockEmbedder.", file=sys.stderr)
        embedding_fn, backend_label = make_embedding_fn("mock")

    if args.all:
        print("=" * 78)
        print("SO SÁNH 4 CHIẾN LƯỢC — cùng corpus, cùng 5 câu hỏi, cùng embedding")
        print(f"Embedding: {backend_label}")
        print("=" * 78)
        rows = []
        for name in ("fixed", "sentence", "recursive", "heading"):
            total, count = run_strategy(name, embedding_fn, backend_label, args.top_k, verbose=False)
            rows.append((name, STRATEGIES[name][0], STRATEGIES[name][1], count, total))
        print(f"\n{'chiến lược':12s} {'cấu hình':34s} {'thành viên':22s} {'chunk':>6s} {'điểm':>6s}")
        for name, label, owner, count, total in sorted(rows, key=lambda r: -r[4]):
            print(f"{name:12s} {label:34s} {owner:22s} {count:6d} {total:4d}/10")
        return 0

    run_strategy(args.strategy, embedding_fn, backend_label, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
