# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** [Đỗ Thanh Tùng]
**Nhóm:** [Top1Server]
**Ngày:** [19/09/2026]

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:*
Hai vector embedding chỉ gần như cùng một hướng trong không gian ngữ nghĩa, nghĩa là hai đoạn văn bản nói về cùng một ý — bất kể chúng dùng từ ngữ khác nhau hay dài ngắn khác nhau. Giá trị chạy từ −1 (ngược hướng) qua 0 (không liên quan) đến 1 (trùng hướng hoàn toàn).
**Ví dụ có độ tương tự CAO:**
- Câu A:"Sinh viên được mượn tối đa 3 tài liệu trong 10 ngày."
- Câu B:"Mỗi bạn học có thể đem về ba cuốn sách và phải trả sau một tuần rưỡi
- Tại sao tương đồng:hai câu gần như không chia sẻ từ vựng nào nhưng mô tả đúng cùng một quy định. Chính vì thế cặp này là phép thử tốt: nếu similarity vẫn cao thì đó là bằng chứng embedding mã hóa ý nghĩa, chứ không phải so khớp chuỗi ký tự. Một cặp chỉ khác nhau vài từ thì không chứng minh được điều gì.

**Ví dụ có độ tương tự THẤP:**
- Câu A:"Phí trễ hạn là 1.000 đồng một tài liệu một ngày."
- Câu B:"Thư viện có bốn tầng, tầng 3 có bốn phòng thảo luận nhóm."
- Tại sao khác:hai câu cùng nói về "thư viện" và dùng chung vài từ, nhưng nói về hai chuyện không liên quan — chế tài tài chính so với bố trí không gian vật lý. Điểm đáng lưu ý: cặp này sẽ cho similarity thấp nhưng không bằng 0, vì cùng miền. Muốn gần 0 thật thì phải lấy câu ngoài miền hẳn, ví dụ "Hôm nay trời Hà Nội mưa nhỏ".

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:*
Vì cosine chỉ đo hướng, bỏ qua độ lớn của vector. Một đoạn văn dài thường sinh vector có norm lớn hơn đoạn ngắn, nên khoảng cách Euclid sẽ phạt oan hai đoạn cùng nghĩa mà khác độ dài — đúng tình huống phổ biến nhất trong retrieval, khi so một câu hỏi ngắn với một chunk dài. Riêng trong repo này, embedding đã được chuẩn hóa về ||v|| = 1, nên dot product bằng đúng cosine — đó là lý do docstring của search cho phép dùng dot product cho gọn.
### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*số chunk = ceil((độ_dài − overlap) / (chunk_size − overlap)) = ceil((10000 − 50) / (500 − 50))

> *Đáp án:*23 chunk

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:*Tăng từ 23 lên 25 chunk (+8,7%). Nguyên nhân: overlap lớn hơn làm bước nhảy nhỏ lại (450 → 400), nên cần nhiều lượt cắt hơn để phủ hết cùng một độ dài văn bản. Muốn độ chồng chéo nhiều hơn vì cắt cố định theo ký tự không quan tâm ranh giới ngữ nghĩa — một câu, một dòng trong bảng, hay một điều khoản có thể bị cắt đôi ngay giữa. Khi đó không chunk nào chứa trọn thông tin và câu trả lời đúng biến mất khỏi kết quả truy xuất. 

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*

Tôi dùng `re.split(r"(?<=[.!?])\s+", text)`. Điểm mấu chốt là **lookbehind** `(?<=...)`: nó chỉ kiểm tra rằng vị trí cắt có dấu câu đứng ngay trước, chứ không tiêu thụ ký tự đó, nên dấu chấm được giữ lại ở cuối mỗi câu. Nếu viết `[.!?]\s+` thì dấu câu bị tính vào phần bị xóa và mọi chunk thành câu cụt — tôi đã chạy thử hai cách để xác nhận khác biệt này. Một biểu thức `\s+` phủ được cả bốn trường hợp `". "`, `"! "`, `"? "`, `".\n"` mà docstring liệt kê, vì `\s` khớp cả dấu cách lẫn xuống dòng.

Sau khi tách, tôi `.strip()` từng câu và loại phần tử rỗng (`if s.strip()`), rồi gom nhóm bằng `range(0, len(sentences), step)` với `step = max_sentences_per_chunk` và nối bằng `" ".join(...)`. Phải `join` chứ không `append` nguyên lát cắt, vì lát cắt của list vẫn là list trong khi test yêu cầu mọi phần tử là `str`.

**Edge case đã xử lý:** text rỗng và text chỉ có khoảng trắng đều trả `[]` — tôi kiểm `if not text or not text.strip()` chứ không chỉ `if not text`, vì `"   \n  "` sẽ lọt qua điều kiện thứ nhất và sinh ra một chunk rỗng.

**Edge case biết là chưa xử lý được:** chữ viết tắt bị cắt sai. Chạy thử trên câu tiếng Việt cho kết quả:

```
'TS.'                      ← bị tách khỏi tên người
'Nguyen Van A phu trach.'
'Cac muc khac v.v.'        ← bị cắt giữa câu
'deu ap dung.'
```

Ngược lại, số phân cách hàng nghìn kiểu Việt Nam (`1.000 đồng`, `5.000đ/tài liệu/ngày` trong corpus của nhóm) **không** bị cắt, vì sau dấu chấm là chữ số chứ không phải khoảng trắng nên `\s+` không khớp. Đây là may mắn của dữ liệu chứ không phải do thiết kế của tôi. Corpus thư viện hiện không chứa `TS.`/`PGS.` nên lỗi này chưa ảnh hưởng, nhưng sẽ hỏng ngay nếu đổi sang tài liệu học vụ có học hàm học vị. Hướng sửa: thêm negative lookbehind loại trừ danh sách viết tắt, hoặc dùng thư viện tách câu chuyên dụng — tôi không làm vì docstring của đề chỉ định rõ cách tách.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?*

`chunk()` chỉ là vỏ mỏng gọi `self._split(text, self.separators)`; toàn bộ logic nằm trong `_split`. Ý tưởng là cắt bằng ranh giới "to" trước để giữ ngữ nghĩa (`"\n\n"` = đoạn văn), chỉ khi mảnh vẫn quá dài mới hạ xuống separator nhỏ hơn theo thứ tự `"\n"` → `". "` → `" "` → `""`.

Thuật toán chạy **hai chiều** trong cùng một vòng lặp:

- *Đệ quy xuống:* mảnh nào dài hơn `chunk_size` thì gọi lại `_split(piece, rest)` với danh sách separator còn lại.
- *Gom lên:* các mảnh nhỏ liền kề được nối vào một `buffer` cho tới sát `chunk_size` rồi mới đẩy ra kết quả. Đây là chiều tôi suýt bỏ quên. Không có nó, `"word " * 200` với `chunk_size=100` sẽ ra **200 chunk 4 ký tự** thay vì 10 chunk 99 ký tự — test vẫn xanh nhưng retrieval sẽ vô dụng vì không chunk nào đủ ngữ cảnh.

Khi nối lại tôi trả separator về đúng chỗ (`buffer + separator + piece`), nếu không văn bản sẽ dính liền mất dấu cách.

**Ba base case:**

1. `current_text` rỗng hoặc chỉ khoảng trắng → `[]`.
2. `len(current_text) <= chunk_size` → `[current_text]`, không cần cắt nữa. Đây là base case dừng đệ quy trong trường hợp thông thường.
3. Hết separator → cắt cứng theo `chunk_size` ký tự một, bỏ qua ranh giới ngữ nghĩa. Nhánh này là bắt buộc vì test `test_empty_separators_falls_back_gracefully` truyền thẳng `separators=[]`.

Một chi tiết dễ sập mà tôi phải xử lý riêng: `DEFAULT_SEPARATORS` kết thúc bằng chuỗi rỗng `""`, mà trong Python `"abc".split("")` ném `ValueError: empty separator`. Nên tôi gộp trường hợp `separator == ""` vào chung nhánh base case 3 thay vì để nó đi qua đường `.split()` thông thường.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*

Store là một `list[dict]` trong bộ nhớ. Quyết định đầu tiên của tôi là **bỏ hẳn nhánh ChromaDB**: không test nào cần nó, `requirements.txt` không cài nó, và code khởi tạo gốc có một cái bẫy — `self._use_chroma = True` được gán *trước* khi client được tạo, nên máy nào tình cờ có `chromadb` sẽ rẽ vào nhánh chưa cài đặt và cả 14 test sập.

Tôi tách hai helper trước khi viết method công khai. `_make_record` chuẩn hoá một `Document` thành record: copy metadata (`dict(doc.metadata or {})`) thay vì dùng thẳng dict của người gọi — nếu không, sửa record sau này sẽ sửa luôn `Document` gốc ở ngoài store; và `metadata.setdefault("doc_id", doc.id)` để `delete_document` luôn có khoá để lọc.

`_search_records` chạy similarity search trên **một tập record bất kỳ**. Tách riêng vì `search()` và `search_with_filter()` chỉ khác nhau ở tập ứng viên đầu vào — cho cả hai đi qua cùng một đường code thì không thể lệch kết quả, và test `test_no_filter_returns_all_candidates` pass hiển nhiên. Độ tương tự tính bằng `_dot(query_embedding, record["embedding"])`: vector đã được chuẩn hoá `||v|| = 1` nên dot product bằng đúng cosine, không cần chia cho norm. Kết quả trả về **bỏ khoá `embedding`** vì vector 64 chiều làm bẩn output khi in ra terminal.

`add_documents` chỉ append — **không tự chunk**. 1 `Document` = 1 record, đúng như test yêu cầu (`get_collection_size() == 3` khi nạp 3 document). Chunking do tầng ngoài (`bench.py`) làm.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*

**Lọc trước, search sau.** Đây là điểm tôi cân nhắc kỹ nhất. Nếu lấy top-k rồi mới bỏ cái không khớp, k slot có thể đã bị chiếm hết bởi tài liệu sai `audience` và hàm trả về 0 kết quả *dù store vẫn còn tài liệu hợp lệ*. Lọc trước thì k slot luôn được lấp bằng ứng viên đúng điều kiện. Điều kiện lọc dùng `all(record["metadata"].get(key) == value for key, value in metadata_filter.items())` — khớp tất cả cặp key-value, dùng `.get()` để record thiếu khoá đó thì trượt chứ không nổ `KeyError`.

Khi `metadata_filter` là `None` hoặc dict rỗng, tôi cho `candidates = self._store` rồi vẫn đi qua `_search_records` như thường — nhờ vậy `search_with_filter(q, metadata_filter=None)` và `search(q)` trả về đúng cùng số kết quả.

`delete_document` dựng lại danh sách bằng list comprehension giữ mọi record có `metadata["doc_id"] != doc_id`, so sánh độ dài trước/sau để biết có xoá được gì không rồi trả `True`/`False`. Xoá theo `doc_id` chứ không theo `Document.id` là có chủ đích: ở `bench.py` một file sinh ra nhiều chunk `"ten-file#0"`, `"ten-file#1"`, và người dùng muốn xoá **cả tài liệu** chứ không phải từng chunk.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*

Ba nhịp: `store.search(question, top_k)` → dựng ngữ cảnh → gọi `llm_fn(prompt)`.

Phần tôi đầu tư nhiều nhất là cách dựng ngữ cảnh. Mỗi chunk được đánh số `[1] [2] [3]` kèm nguồn lấy từ `metadata["source_url"]` (fallback sang `doc_id` rồi `id`), và prompt yêu cầu model trích dẫn đúng số đó khi trả lời. Nhờ vậy câu trả lời **truy vết được** về đúng chunk và đúng file — đây là tiêu chí *Source Traceability* trong `docs/EVALUATION.md`, và với corpus quy định thì nó không phải tính năng phụ: người đọc phải kiểm được con số "3 tài liệu / 10 ngày" đến từ văn bản nào.

Prompt có bốn ràng buộc chống bịa: chỉ dùng ngữ cảnh được cung cấp, trích dẫn số nguồn, nói rõ khi không đủ thông tin, và giữ nguyên con số/mốc thời gian/đơn vị đúng như trong ngữ cảnh. Ràng buộc cuối quan trọng với dữ liệu quy định vì model rất hay làm tròn hoặc đổi đơn vị.

Trường hợp store rỗng (hoặc không truy xuất được gì) thì trả thẳng một câu thông báo — **không gọi LLM**, vì gọi mà không có ngữ cảnh chỉ tạo cơ hội cho model bịa.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.10.4, pytest-9.1.1, pluggy-1.6.0 -- E:\K4-Day07-DoThanhTung-2A202602845\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\K4-Day07-DoThanhTung-2A202602845
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.04s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tôi đo **hai lần trên cùng 5 cặp câu**, một lần bằng `MockEmbedder` và một lần bằng `LocalEmbedder` (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều). So sánh hai cột là phần thú vị nhất của bài tập này.

| Cặp | Câu A | Câu B | Dự đoán | MockEmbedder | LocalEmbedder | Đúng? |
|------|-----------|-----------|---------|--------------|---------------|-------|
| 1 | Sinh viên được mượn tối đa 3 tài liệu trong 10 ngày. | Mỗi bạn học có thể đem về ba cuốn sách và trả sau một tuần rưỡi. | cao | −0,1748 | **+0,6598** | ✓ |
| 2 | Phí trễ hạn là 1.000 đồng một tài liệu một ngày. | Trả sách muộn sẽ bị phạt tiền theo số ngày quá hạn. | cao | −0,1678 | **+0,5094** | ✓ |
| 3 | Thẻ thư viện cấp mới có lệ phí 100.000 đồng. | Lệ phí làm thẻ thư viện lần đầu là một trăm nghìn đồng. | cao | +0,0189 | **+0,7650** | ✓ |
| 4 | Phòng học nhóm ở tầng 3 chứa được 5 đến 7 người. | Người sử dụng phải đền gấp 5 lần giá bìa nếu làm mất sách. | thấp | +0,0280 | **+0,2675** | ✓ |
| 5 | Thư viện mở cửa từ 7 giờ sáng đến 8 giờ tối. | Hôm nay trời Hà Nội mưa nhỏ và se lạnh. | thấp | +0,0123 | **+0,1019** | ✓ |

**Đúng 5/5 với `LocalEmbedder`, 0/5 về mặt thứ hạng với `MockEmbedder`.**

Ba cặp đầu tôi cố ý chọn **khác từ vựng nhưng cùng nghĩa** — đó mới là phép thử thật cho embedding, vì một cặp chỉ khác nhau vài từ thì không chứng minh được gì. Cặp 4 cùng miền thư viện nhưng khác chủ đề, cặp 5 khác miền hẳn.

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*

Bất ngờ nhất là cột `MockEmbedder`: **ba cặp cùng nghĩa lại cho điểm thấp nhất bảng, hai cặp còn âm** (−0,17), thấp hơn cả cặp "Thư viện mở cửa 7 giờ" với "Hôm nay Hà Nội mưa" là hai câu không liên quan gì nhau. Thứ hạng bị đảo ngược hoàn toàn. `MockEmbedder` băm MD5 chuỗi ký tự rồi sinh dãy số giả ngẫu nhiên, nên hai câu chỉ khác một dấu phẩy cũng cho ra hai vector hoàn toàn khác. Nó thoả mãn *hình thức* của một embedder — vector cùng chiều, đã chuẩn hoá, `compute_similarity` chạy đúng công thức — nhưng thiếu đúng tính chất cốt lõi: **câu gần nghĩa thì vector gần nhau**. Hai dòng "đúng" ở cặp 4 và 5 là trúng ngẫu nhiên.

Điều bất ngờ thứ hai nằm ở cột `LocalEmbedder`: **cặp 3 đạt 0,7650, cao hơn cặp 1 (0,6598)** dù cặp 1 mới là cặp tôi cố tình viết khác từ vựng nhất. Lý do là cặp 3 vẫn giữ cụm "lệ phí" và "thẻ thư viện" giống nhau, chỉ đổi "100.000 đồng" thành "một trăm nghìn đồng"; còn cặp 1 đổi gần như toàn bộ từ vựng (*sinh viên/bạn học*, *mượn/đem về*, *3/ba*, *10 ngày/một tuần rưỡi*). Nghĩa là embedding **có** nắm ngữ nghĩa — 0,66 vẫn là điểm cao — nhưng độ trùng từ vựng vẫn đẩy điểm lên thêm. Nó không phải bộ hiểu nghĩa thuần tuý.

Quan sát thứ ba: cặp 4 đạt 0,2675, gấp 2,6 lần cặp 5 (0,1019), đúng như tôi đoán khi viết chúng — hai câu cùng miền thư viện thì không bao giờ về 0, chỉ câu ngoài miền hẳn mới xuống gần 0. Đây chính là lý do trong retrieval thật, ngưỡng similarity tuyệt đối gần như vô dụng: mọi chunk trong cùng một corpus đều "hơi giống" câu hỏi, chỉ có **thứ hạng tương đối** mới nói lên điều gì.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Cấu hình đo:** corpus `data/thu-vien-huit/` (8 tài liệu), chiến lược của tôi là **`FixedSizeChunker(chunk_size=500, overlap=50)`**, 63 chunk, embedding `LocalEmbedder` (`paraphrase-multilingual-MiniLM-L12-v2`). Công cụ đo: `bench.py`. Output đầy đủ: `ket_qua_benchmark.txt`.

**Lý do chọn:** corpus của nhóm là văn bản quy định có **mật độ số liệu cao** — hạn mức, lệ phí, mức phạt, thời hạn. Thứ tôi sợ nhất là một con số bị cắt rời khỏi ngữ cảnh giải thích nó, ví dụ "100.000 đ/thẻ" nằm ở chunk này còn "Thẻ cấp mới" ở chunk kia. `overlap=50` mua bảo hiểm cho đúng rủi ro đó: thông tin nằm gần ranh giới xuất hiện trong **hai** chunk liền nhau, nên có hai cơ hội lọt top-k thay vì một.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Được mượn tối đa bao nhiêu tài liệu và trong bao nhiêu ngày? *(lọc `audience=student`)* | `han-muc-muon-tai-lieu-sinh-vien#0` — tiêu đề + bảng hạn mức của sinh viên | +0,6956 | **Có** — 2/3 slot là gold doc | Ngữ cảnh chứa "Sinh viên, học viên \| 3 \| 10" → trả lời đúng: 3 tài liệu, 10 ngày |
| 2 | Điều kiện để được mượn tài liệu về nhà? | `han-muc-muon-tai-lieu-sinh-vien#4` — đoạn xử lý trả quá hạn | +0,7299 | **Một phần** — gold doc ở top-3 nhưng chunk bị cắt trước con số | Ngữ cảnh thiếu "25/35" → agent nêu được "tham gia lớp tập huấn" nhưng **mất điều kiện định lượng** |
| 3 | Quy trình đặt phòng học nhóm và mốc hủy khi đến trễ? | `quy-dinh-su-dung-thu-vien#20` — mục 9, quy định đặt phòng chung | +0,6371 | **Không** — gold doc `su-dung-phong-hoc-nhom` vắng khỏi top-3 | Ngữ cảnh có "ĐẶT PHÒNG" và "trễ trên 15 phút" nhưng **từ tài liệu khác**, thiếu 3 bước Lầu 3/Lầu 4 |
| 4 | Mượn liên thư viện: số lượng, thời hạn, phí trễ hạn? | `muon-lien-thu-vien#3` — mục "Quy định" | +0,8043 | **Có** — gold doc ở top-1, điểm cao nhất cả bộ | Ngữ cảnh chứa đủ cả 3 con số → trả lời đúng hoàn toàn |
| 5 | Lệ phí thẻ và thời gian trả thẻ? | `huong-dan-su-dung-thu-vien#5` — bảng lệ phí + thời gian trả thẻ | +0,7200 | **Có** — gold doc ở top-1 | Ngữ cảnh chứa "100.000 đ/thẻ" và "07 ngày kể từ ngày đăng ký" → trả lời đúng |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4** / 5 (chỉ câu 3 vắng hẳn gold doc). Nhưng theo thang chấm nghiêm thì câu 2 vẫn 0đ vì ngữ cảnh thiếu con số. **Tổng điểm retrieval: 6/10** (câu 1, 4, 5 được 2đ; câu 2, 3 được 0đ).

### Vì sao tôi chấm hai mức chứ không chỉ kiểm `doc_id`

`bench.py` của tôi khai báo cho mỗi câu **một danh sách chuỗi đặc trưng** (`must_contain`) phải **cùng** xuất hiện trong ngữ cảnh truy xuất được, rồi kiểm từng chuỗi có thật không.

Câu 2 là ví dụ đắt giá nhất. `luu-hanh-tai-lieu` **có** nằm ở top-3 (hạng 3, score 0,6929), nếu chấm bằng `doc_id` thì đây là câu "đạt". Nhưng chunk được lấy bắt đầu bằng *"ng Vebrary có trên App Store và Google Play. Điều kiện sử dụng: đã tham gia lớp tập huấn…"* — nó dừng lại **ngay trước** dòng chứa "đạt 25/35 câu". Agent sẽ trả lời "cần tham gia lớp tập huấn" mà không nêu được ngưỡng điểm, tức là trả lời thiếu đúng phần định lượng mà câu hỏi nhắm tới. Chênh lệch **4/5 so với 3/5** chính là mức thổi phồng của cách chấm ngây thơ.

Câu 3 cho thấy mặt ngược lại: ngữ cảnh **có đủ** cả "ĐẶT PHÒNG" lẫn "trễ trên 15 phút", nhưng chúng đến từ `quy-dinh-su-dung-thu-vien#20` chứ không phải tài liệu gold. Hai tài liệu này mô tả cùng một quy trình đặt phòng bằng từ ngữ gần như trùng nhau, nên retrieval lấy nhầm là dễ hiểu. Tôi vẫn chấm 0đ vì gold doc vắng mặt — nhưng thực tế agent **vẫn trả lời được phần lớn câu hỏi**. Đây là giới hạn của việc gắn một câu hỏi với đúng một tài liệu gold khi corpus có nội dung chồng lấn.

### A/B metadata filter (câu 1 — câu đánh bẫy)

| | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| **Không filter** | `muon-lien-thu-vien#3` (0,7505) | `han-muc-muon-tai-lieu-sinh-vien#0` (0,6956) | `han-muc-muon-tai-lieu-giang-vien#0` (0,6866) |
| **Có `audience=student`** | `han-muc-muon-tai-lieu-sinh-vien#0` (0,6956) | `han-muc-muon-tai-lieu-sinh-vien#4` (0,6769) | `huong-dan-su-dung-thu-vien#5` (0,5521) |

Đây đúng là một cái bẫy hoạt động. Không lọc thì ngữ cảnh gửi cho LLM chứa **ba con số khác nhau cho cùng một câu hỏi**: 2 tài liệu/20 ngày (liên thư viện, top-1), 3 tài liệu/10 ngày (sinh viên, top-2), 3 tài liệu/180 ngày (giảng viên, top-3). Con số sai lại có score **cao nhất**. Agent gần như chắc chắn trả lời sai đối tượng, hoặc tệ hơn là trộn lẫn ba mức hạn mức.

Bật filter thì tài liệu của giảng viên và liên thư viện bị loại sạch ngay từ vòng ứng viên, chỉ còn tài liệu đúng đối tượng.

Lưu ý điểm số **giảm** khi bật filter (0,7505 → 0,6956). Đúng như kỳ vọng: filter không làm chunk nào giống câu hỏi hơn, nó chỉ loại ứng viên sai đối tượng để nhường slot. Điểm thấp hơn nhưng câu trả lời đúng — bằng chứng rõ ràng rằng **score cao không đồng nghĩa với retrieval tốt**.

### So sánh 4 chiến lược trên cùng bộ câu hỏi

| Chiến lược | Số chunk | Q1 | Q2 | Q3 | Q4 | Q5 | Tổng |
|---|---|---|---|---|---|---|---|
| `RecursiveChunker` (500) | 83 | 2 | 0 | 2 | 2 | 2 | **8/10** |
| `HeadingChunker` (900, custom) | 60 | 2 | 2 | 1 | 2 | 0 | 7/10 |
| **`FixedSizeChunker` (500/50) — của tôi** | 63 | 2 | 0 | 0 | 2 | 2 | **6/10** |
| `SentenceChunker` (4 câu) | 63 | 2 | 0 | 0 | 0 | 0 | 2/10 |

Chiến lược của tôi xếp thứ ba. Ba quan sát tôi rút ra:

**Overlap giúp ở câu tra bảng.** Câu 5 hỏi lệ phí thẻ, chunk `huong-dan-su-dung-thu-vien#5` bắt đầu giữa chừng bảng (`| Thẻ cấp lại | Phí gia hạn/năm |…`) nhưng vẫn ôm trọn cả "100.000 đ/thẻ" lẫn "07 ngày kể từ ngày đăng ký" nhờ overlap 50 — hai thông tin cách nhau khá xa trong văn bản gốc. `HeadingChunker` bị 0đ đúng ở câu này vì nó tách bảng lệ phí và mục thời gian trả thẻ thành hai chunk khác nhau.

**Nhưng cắt cứng phá hỏng câu hỏi quy trình.** Câu 3 cần cả 3 bước đăng ký lẫn mốc "trễ 15 phút" — `fixed` xé quy trình làm đôi nên không chunk nào đủ, gold doc rơi khỏi top-3. `RecursiveChunker` giữ được vì nó cắt theo ranh giới `\n\n` trước, tôn trọng cấu trúc đoạn.

**`SentenceChunker` sụp đổ ở 2/10.** Với văn bản quy định viết dưới dạng bảng Markdown và danh sách gạch đầu dòng, tách theo dấu câu gần như không có ranh giới để bám — một bảng 6 dòng không có dấu chấm nào nên bị gom thành một chunk khổng lồ, còn các gạch đầu dòng ngắn thì bị trộn lẫn tùy tiện. Đây là bài học rõ nhất: **chiến lược chunking phải khớp với cấu trúc thật của tài liệu**, không có lựa chọn mặc định tốt cho mọi loại văn bản.

### Ghi chú về câu trả lời của Agent

Repo không cấu hình API key LLM nào, nên `llm_fn` là hàm `demo_llm` trong `main.py` — nó chỉ in lại đầu prompt chứ không sinh câu trả lời thật. Cột "Câu trả lời của Agent" ở bảng trên vì vậy phản ánh **ngữ cảnh mà agent nhận được có chứa đáp án hay không**, tức là chất lượng khâu truy xuất và dựng prompt, chứ không phải chất lượng sinh văn bản. Với một LLM thật, câu 1 và câu 4 sẽ trả lời đúng vì đáp án nằm sẵn trong ngữ cảnh; câu 2, 3, 5 vẫn sai vì ngữ cảnh không chứa thông tin cần — và prompt của tôi bắt model nói rõ là không tìm thấy thay vì bịa.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*

*(điền sau buổi demo)*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 6 / 10 |
| **Tổng phần cá nhân** | **56 / 60** |

Căn cứ tự chấm:

- **Hoàn thiện code 30/30** — `pytest tests/ -v` cho 42 passed, không còn `raise NotImplementedError` nào trong `src/`.
- **Kết quả truy xuất 6/10** — chấm theo đúng thang trong `docs/SCORING.md` (2đ/câu): câu 1, 4, 5 được 2đ; câu 2 và 3 được 0đ. Tôi tự chấm theo cách nghiêm, yêu cầu ngữ cảnh chứa **đủ** mọi chuỗi đặc trưng của gold answer chứ không chỉ kiểm `doc_id` có trong top-3 — cách chấm lỏng sẽ cho 4/5 câu.
- **Khởi động 5/5** — đã kiểm lại phép tính chunking bằng chính `FixedSizeChunker` trong repo (23 chunk, khớp công thức) thay vì chỉ tin công thức.
- **Dự đoán độ tương tự 5/5** — đo hai lần trên hai backend, đúng 5/5 với embedder thật, và phân tích được vì sao `MockEmbedder` cho kết quả đảo ngược.
