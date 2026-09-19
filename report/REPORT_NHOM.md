# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Top1Server
**Thành viên:** Đỗ Thanh Tùng (R1), Phạm Đức Anh (R2), Trần Võ Hoàng Nguyên (R3), Ninh Quang Minh (R4)
**Ngày:** 19/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

**Cấu hình đo dùng chung cho mọi số liệu trong báo cáo này:**

| Hạng mục | Giá trị |
|---|---|
| Corpus | `data/thu-vien-huit/` — 8 tài liệu, 27.504 ký tự nội dung |
| Embedding | `LocalEmbedder` — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, **không cần API key** |
| Công cụ đo | `bench.py` (chạy `python bench.py --all` để tái lập bảng so sánh) |
| top-k | 3 |
| Output thô | `ket_qua_benchmark.txt` |

Nhóm thống nhất dùng `LocalEmbedder` thay vì `MockEmbedder`. Lý do: `MockEmbedder` băm MD5 nội dung nên **không mã hoá ngữ nghĩa** — chạy thử bằng nó cho `0/10` ở mọi chiến lược, mọi số liệu đều là nhiễu. Cũng không dùng OpenAI vì `requirements.txt` không cài `openai` và cần API key, người khác clone repo về sẽ không chạy được.

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ và quy định thư viện đại học — Trung tâm Thông tin – Thư viện, Trường ĐH Công Thương TP.HCM (`thuvien.huit.edu.vn`).

**Tại sao nhóm chọn chủ đề này?**

K4-L3A bắt buộc chủ đề "dịch vụ hoặc quy định đại học". Trong các mảng được phép, thư viện là mảng nhóm tìm được nguồn vừa **cho phép truy cập tự động** vừa đủ 5–10 trang chất lượng. Nhóm đã thử và **loại 6 nguồn**: `lib.hnue.edu.vn`, `glib.hcmus.edu.vn`, `tlu.edu.vn`, `huflit.edu.vn`, `hcmus.edu.vn` bị `robots.txt` cấm; `lic.ut.edu.vn` trả HTTP 403 cho client tự động. Theo `docs/DATA_COLLECTION.md` mục 2, 403 là tín hiệu không cho crawl nên nhóm đổi nguồn thay vì giả `User-Agent` trình duyệt để lách.

Yếu tố quyết định là trang quy định của HUIT có **bảng hạn mức mượn theo từng đối tượng**: sinh viên 3 tài liệu/10 ngày, giảng viên 3 tài liệu/180 ngày, giảng viên thỉnh giảng 2/30, người ngoài trường 1/10. Đây chính là thứ `docs/DATA_COLLECTION.md` mục 4 yêu cầu tách file theo `audience`, và là điều kiện để `metadata_filter` có việc thật thay vì chỉ đẹp trên giấy.

### Danh sách tài liệu (Data Inventory)

| # | doc_id | Tiêu đề | URL gốc | retrieved_at / version | Số ký tự | Metadata đã gán |
|---|---|---|---|---|---|---|
| 1 | `quy-dinh-su-dung-thu-vien` | Quy định sử dụng thư viện HUIT | `/Page/quy-dinh-su-dung-thu-vien` | 2026-09-19 / not-stated | 11.209 | audience=all, department=library, category=regulation, language=vi |
| 2 | `huong-dan-su-dung-thu-vien` | Hướng dẫn sử dụng thư viện | `/Page/huong-dan-su-dung-thu-vien` | 2026-09-19 / not-stated | 4.943 | audience=student, category=guide |
| 3 | `luu-hanh-tai-lieu` | Dịch vụ lưu hành tài liệu | `/Page/luu-hanh-tai-lieu` | 2026-09-19 / not-stated | 3.074 | audience=all, category=circulation |
| 4 | `muon-lien-thu-vien` | Dịch vụ mượn liên thư viện | `/Page/muon-lien-thu-vien` | 2026-09-19 / not-stated | 2.209 | audience=all, category=interlibrary-loan |
| 5 | `han-muc-muon-tai-lieu-sinh-vien` | Hạn mức mượn tài liệu của sinh viên | `/Page/quy-dinh-su-dung-thu-vien` | 2026-09-19 / not-stated | 1.902 | **audience=student**, category=borrowing-limit |
| 6 | `su-dung-phong-hoc-nhom` | Quy định sử dụng phòng học nhóm | `/Page/su-dung-phong-hoc-nhom` | 2026-09-19 / not-stated | 1.748 | audience=student, category=facility |
| 7 | `han-muc-muon-tai-lieu-giang-vien` | Hạn mức mượn tài liệu của giảng viên | `/Page/quy-dinh-su-dung-thu-vien` | 2026-09-19 / not-stated | 1.336 | **audience=faculty**, category=borrowing-limit |
| 8 | `muon-tra-sach-tu-dong` | Hướng dẫn mượn trả sách tự động | `/Page/muon-tra-sach-tu-dong` | 2026-09-19 / not-stated | 1.083 | audience=all, category=self-service |

Tiền tố URL: `https://thuvien.huit.edu.vn`. `sources.csv` khớp 1-1 với 8 file.

Tài liệu 5 và 7 **cùng `source_url` với tài liệu 1** — đó là chủ ý. Trang quy định gộp hạn mức của cả sinh viên lẫn giảng viên trong một bảng; nếu để nguyên một file `audience: all` thì `metadata_filter={"audience":"student"}` không lọc được gì vì hai đáp án nằm chung một tài liệu. Nhóm tách thành hai file, mỗi file một `audience`, đúng hướng dẫn `docs/DATA_COLLECTION.md` mục 4.

**Về `document_version = not-stated`:** không trang nguồn nào công bố số hiệu quyết định hay ngày hiệu lực. Nhóm ghi `not-stated` thay vì bịa số hiệu. Hệ quả: corpus này không kiểm thử được kịch bản "tài liệu cũ vs mới".

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu chỉ chứa nguồn công khai/được phép dùng, không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ. Toàn bộ là trang công khai, không cần đăng nhập. `license_or_permission = public-source` cho cả 8 tài liệu.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` trong metadata. Crawl bằng `scripts/fetch_public_pages.py` với `--delay 2` (gấp đôi mức tối thiểu 1 giây), kiểm `robots.txt` trước mỗi request, sau đó làm sạch thủ công (gỡ menu/footer, dựng lại bảng Markdown).

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|---|---|---|---|
| `doc_id` | string | `han-muc-muon-tai-lieu-sinh-vien` | Khoá để `delete_document()` xoá cả tài liệu, và là khoá đối chiếu gold answer khi chấm. Trỏ về **tên file gốc**, không phải id chunk. |
| `audience` | enum: student/faculty/staff/all | `student` | **Trường lọc chính.** Corpus có hai tài liệu cùng chủ đề, cùng từ vựng, khác đối tượng và khác đáp án — không lọc thì agent trả lời sai đối tượng. Bằng chứng A/B ở mục 3. |
| `category` | string, 8 giá trị | `borrowing-limit` | Thu hẹp theo loại nội dung khi câu hỏi rõ ý định: `regulation` vs `guide` vs `facility`. Chưa dùng trong bài này nhưng là hướng sửa cho failure case ở mục 4. |
| `department` | string | `library` | Hiện là hằng số. Giữ lại để corpus mở rộng sang phòng Đào tạo, CTSV mà không phải đổi schema. |
| `language` | string | `vi` | Lọc ngôn ngữ khi trộn tài liệu song ngữ. Hiện là hằng số. |
| `source_url` | url | `https://thuvien.huit.edu.vn/Page/...` | Truy vết nguồn. `KnowledgeBaseAgent` in trường này kèm số `[1] [2] [3]` trong ngữ cảnh để câu trả lời trích dẫn được — tiêu chí *Source Traceability* trong `docs/EVALUATION.md`. |
| `retrieved_at` | date | `2026-09-19` | Kiểm tra độ mới của dữ liệu. |
| `document_version` | string | `not-stated` | Phân biệt phiên bản quy định khi nguồn có công bố. |

Phân bố `audience`: `all` × 4, `student` × 3, `faculty` × 1 — đạt yêu cầu tối thiểu 2 giá trị khác nhau.

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

Mỗi thành viên thử một chiến lược khác nhau trên cùng bộ tài liệu, cùng 5 câu hỏi, cùng embedding backend. Chỉ đổi **một dòng** chọn chunker để so sánh được công bằng.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy trên 2 tài liệu đại diện, **đã bỏ front matter** trước khi đo (nếu không là đang đo cả khối YAML).

**Tài liệu 1: `quy-dinh-su-dung-thu-vien.md`** — 11.209 ký tự, 11 heading `##` + 4 heading `###`

| Chiến lược | Số chunk | Avg | Min | Max | Giữ ngữ cảnh? |
|---|---|---|---|---|---|
| FixedSizeChunker (500, overlap 50) | 25 | 496 | 409 | 500 | ❌ Cắt ngang giữa câu và giữa bảng |
| SentenceChunker (max 3 câu) | 32 | 349 | 107 | 715 | ⚠️ Giữ ranh giới câu nhưng chunk vượt 500 (max 715) |
| RecursiveChunker (500) | 36 | 310 | 30 | 499 | ✅ Phần lớn ≤ 500; có chunk rất nhỏ 30 ký tự ở đầu mục |
| HeadingChunker (max 800, min 200) | 21 | 542 | 32 | 832 | ✅ Giữ trọn từng mục quy định, tự gắn lại tiêu đề khi tách nhỏ |

**Tài liệu 2: `huong-dan-su-dung-thu-vien.md`** — 4.943 ký tự, có bảng giờ mở cửa 4 tầng

| Chiến lược | Số chunk | Avg | Min | Max | Giữ ngữ cảnh? |
|---|---|---|---|---|---|
| FixedSizeChunker (500, overlap 50) | 11 | 495 | 443 | 500 | ❌ Cắt bảng Markdown giữa chừng |
| SentenceChunker (max 3 câu) | 15 | 328 | 85 | **1.411** | ❌ Bảng không có dấu chấm câu → chunk phình to |
| RecursiveChunker (500) | 15 | 328 | 247 | 465 | ✅ Tách đúng newline, kích thước ổn định |
| HeadingChunker (max 800, min 200) | 10 | 509 | 82 | 818 | ✅ Bảng tiện ích các tầng nằm trọn trong một chunk |

**Tổng số chunk trên toàn bộ 8 file:**

| Chiến lược | Tổng chunk |
|---|---|
| HeadingChunker (max 800, min 200) | 56 |
| FixedSizeChunker (500, overlap 50) | 63 |
| RecursiveChunker (500) | 83 |
| SentenceChunker (max 3 câu) | 84 |

**Nhận xét baseline:**

- `FixedSizeChunker` đều và dự đoán được (avg 495–496, sát `chunk_size`), nhưng cắt máy móc: một bảng hạn mức bị chia đôi là mất ý nghĩa tra cứu. Overlap 50 là điểm bù duy nhất — thông tin gần ranh giới xuất hiện ở hai chunk liền nhau.
- `SentenceChunker` hợp với văn xuôi, nhưng bảng và danh sách gạch đầu dòng không kết thúc bằng dấu chấm nên sinh chunk 1.411 ký tự. Đây là chiến lược lệch nhất với cấu trúc corpus.
- `RecursiveChunker` là baseline tốt nhất trong ba chiến lược có sẵn: ưu tiên cắt theo `\n\n` rồi `\n`, giữ kích thước tương đối đều. Hạn chế: không nhận biết heading nên một mục quy định có thể bị tách khỏi tiêu đề.
- `HeadingChunker` cho **ít chunk nhất (56)** với avg lớn nhất — nó khai thác cấu trúc heading có sẵn của văn bản hành chính, gộp section ngắn để tránh chunk rác, và gắn lại tiêu đề khi section vượt ngưỡng.

### Chiến lược của từng thành viên

| Thành viên | Vai | Chiến lược |
|---|---|---|
| Đỗ Thanh Tùng | R1 · Data | `FixedSizeChunker(500, overlap 50)` |
| Phạm Đức Anh | R2 · Benchmark | `RecursiveChunker(500)` |
| Trần Võ Hoàng Nguyên | R3 · Strategy | `HeadingChunker(max 800, min 200)` — **vai chunk theo heading bắt buộc của K4-L3A** |
| Ninh Quang Minh | R4 · Report & Demo | `SentenceChunker(max 3 câu)` |

**Đỗ Thanh Tùng — FixedSizeChunker (500, overlap 50)**

Corpus là văn bản quy định có **mật độ số liệu cao** — hạn mức, lệ phí, mức phạt, thời hạn. Rủi ro lớn nhất là một con số bị cắt rời khỏi nhãn giải thích nó ("100.000 đ/thẻ" ở chunk này, "Thẻ cấp mới" ở chunk kia). `overlap=50` mua bảo hiểm cho đúng rủi ro đó. Đây cũng là chiến lược duy nhất trong ba chiến lược có sẵn tạo ra dư thừa có chủ đích.

**Phạm Đức Anh — RecursiveChunker (500)**

Thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]` — cắt bằng ranh giới "to" trước để giữ ngữ nghĩa, chỉ khi mảnh vẫn quá dài mới hạ xuống separator nhỏ hơn. Phù hợp với văn bản quy định vì tôn trọng ranh giới đoạn và danh sách gạch đầu dòng mà không cần biết trước cấu trúc heading.

**Trần Võ Hoàng Nguyên — HeadingChunker (custom)**

Văn bản quy định được người soạn chia sẵn theo mục (`## 5. Quy định mượn/trả tài liệu`), mỗi mục đã là một đơn vị ngữ nghĩa trọn vẹn. Thuật toán ba bước: cắt trước dòng heading; gộp section ngắn hơn `min_chars` vào section liền trước để tránh chunk rác; section dài hơn `max_chars` thì hạ xuống `RecursiveChunker` và **gắn lại tiêu đề vào từng mảnh con**. Không có bước cuối thì mảnh thứ hai trở đi mất ngữ cảnh "đây là mục nói về cái gì".

```python
class HeadingChunker:
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
            if (merged and len(merged[-1]) < self.min_chars
                    and len(merged[-1]) + 2 + len(section) <= self.max_chars):
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
```

**Ninh Quang Minh — SentenceChunker (max 3 câu)**

Tách theo ranh giới câu bằng `re.split(r"(?<=[.!?])\s+", text)` — lookbehind giữ lại dấu câu thay vì nuốt mất — rồi gom 3 câu liền nhau thành một chunk. Giả thuyết ban đầu: văn bản quy định viết thành câu hoàn chỉnh nên tách theo câu sẽ giữ được ý trọn vẹn. Kết quả thực tế bác bỏ giả thuyết này (xem bảng dưới).

### So Sánh Giữa Các Thành Viên

Điểm từng câu, chấm theo `docs/SCORING.md` (2đ/câu):

| Thành viên | Chiến lược | Chunk | Q1 | Q2 | Q3 | Q4 | Q5 | **Tổng** |
|---|---|---|---|---|---|---|---|---|
| Phạm Đức Anh | RecursiveChunker (500) | 83 | 2 | 0 | 2 | 2 | 2 | **8/10** |
| Trần Võ Hoàng Nguyên | HeadingChunker (800/200) | 56 | 2 | 2 | 1 | 2 | 0 | **7/10** |
| Đỗ Thanh Tùng | FixedSizeChunker (500/50) | 63 | 2 | 0 | 0 | 2 | 2 | **6/10** |
| Ninh Quang Minh | SentenceChunker (max 3) | 84 | 2 | 0 | 0 | 0 | 2 | **4/10** |

| Thành viên | Điểm mạnh | Điểm yếu |
|---|---|---|
| Đức Anh | Tôn trọng ranh giới đoạn nên giữ trọn quy trình nhiều bước; là người duy nhất được trọn điểm câu 3 cùng Nguyên | Tạo nhiều chunk (83), tốn embedding; có chunk vụn 30 ký tự |
| Nguyên | Chunk dễ đọc và truy vết nhất — mỗi chunk là một mục có tiêu đề; là người **duy nhất** trả lời được câu 2 | Không chồng lấn, mỗi thông tin chỉ có một cơ hội lọt top-k; thua câu 5 vì tách bảng lệ phí khỏi mục thời gian trả thẻ |
| Tùng | Overlap 50 cứu được câu tra bảng (câu 5) khi hai thông tin cách xa nhau trong văn bản gốc | Cắt máy móc giữa câu, phá hỏng câu hỏi quy trình (câu 3) |
| Minh | Chunk đọc tự nhiên nhất với phần văn xuôi | Sụp đổ với bảng biểu: chunk lên tới 1.411 ký tự vì không có dấu câu để bám |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**

`RecursiveChunker` thắng với 8/10, và nhóm cho rằng đây không phải may rủi. Lý do nằm ở **cấu trúc thật của corpus**: văn bản quy định của thư viện được viết chủ yếu dưới dạng đoạn ngắn, danh sách gạch đầu dòng và bảng Markdown — tức là ranh giới ngữ nghĩa nằm ở ký tự xuống dòng, không nằm ở dấu chấm. `RecursiveChunker` ưu tiên `"\n\n"` rồi `"\n"` nên bám đúng vào ranh giới đó, trong khi `SentenceChunker` đi tìm dấu chấm ở nơi gần như không có (4/10) và `FixedSizeChunker` bỏ qua ranh giới hoàn toàn (6/10).

Nhưng kết luận quan trọng hơn là **không chiến lược nào thắng mọi câu hỏi**, và ba chiến lược thắng ở ba loại câu khác nhau:

- **Câu 2** (hỏi điều kiện, đáp án là một dòng nằm cuối một mục): chỉ `HeadingChunker` được 2đ — nó giữ trọn mục nên dòng "đạt 25/35 câu" không bị rơi ra ngoài.
- **Câu 3** (hỏi quy trình nhiều bước + một mốc thời gian): `RecursiveChunker` 2đ, `HeadingChunker` 1đ, hai chiến lược còn lại 0đ.
- **Câu 5** (tra bảng giá + thời hạn, hai thông tin cách xa nhau): `FixedSizeChunker`, `RecursiveChunker`, `SentenceChunker` được 2đ, `HeadingChunker` 0đ.

Nếu phải chọn một chiến lược để triển khai thật cho corpus quy định, nhóm sẽ chọn **`HeadingChunker` cộng overlap** — lấy ưu điểm truy vết của heading (mỗi chunk là một mục có tiêu đề, agent trích dẫn `[1]` là biết ngay mục nào) và bù đúng điểm yếu duy nhất của nó bằng cách cho các section liền kề chồng lấn một phần. Nhóm chưa kịp cài đặt và đo thử phương án này.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|---|---|---|
| 1 — **CÂU ĐÁNH BẪY** | Được mượn tối đa bao nhiêu tài liệu và trong bao nhiêu ngày? | Sinh viên, học viên được mượn tối đa 3 tài liệu trong 10 ngày. Bắt buộc dùng `metadata_filter={"audience": "student"}`. | `han-muc-muon-tai-lieu-sinh-vien.md` — "Hạn mức mượn về nhà". Chạy A/B một lần không filter, một lần có filter. |
| 2 | Người sử dụng cần đáp ứng những điều kiện nào để được mượn tài liệu về nhà? | Hoàn thành bài kiểm tra hướng dẫn sử dụng thư viện đạt 25/35 câu, đăng ký thẻ thư viện và đóng tiền thế chân theo quy định. | `luu-hanh-tai-lieu.md` — "Dịch vụ cho mượn về / Điều kiện sử dụng". |
| 3 | Quy trình đăng ký và sử dụng phòng học nhóm gồm những bước nào, và đến trễ bao lâu thì kết quả đặt phòng bị hủy? | Đăng ký tại Quầy thông tin hoặc mục ĐẶT PHÒNG trực tuyến; nhận kết quả qua email; đến Quầy thông tin Lầu 3 hoặc Lầu 4 làm thủ tục. Đến trễ trên 15 phút thì kết quả bị hủy. | `su-dung-phong-hoc-nhom.md` — "Hướng dẫn sử dụng" và "Lưu ý". |
| 4 | Dịch vụ mượn liên thư viện cho phép mượn tối đa bao nhiêu tài liệu, trong bao lâu và phí trễ hạn là bao nhiêu? | Tối đa 2 tài liệu/lần, thời hạn 20 ngày từ ngày nhận, phí trễ hạn 5.000 đồng/tài liệu/ngày. | `muon-lien-thu-vien.md` — "Quy định". |
| 5 | Lệ phí cấp mới, cấp lại, gia hạn thẻ thư viện là bao nhiêu và thời gian trả thẻ được quy định thế nào? | Cấp mới 100.000 đồng, cấp lại 50.000 đồng, gia hạn 50.000 đồng/năm; thẻ mới trả sau bài kiểm tra 1 tuần hoặc theo lịch hẹn, thẻ cấp lại sau 7 ngày. | `huong-dan-su-dung-thu-vien.md` — "Đăng ký làm thẻ". |

Năm câu phủ năm dạng hỏi khác nhau: tra số liệu cần lọc metadata (1), hỏi điều kiện (2), hỏi quy trình nhiều bước (3), tra số liệu ghép ba con số (4), tra bảng giá kèm thời hạn (5).

### Cách nhóm chấm — hai mức, không chỉ kiểm `doc_id`

Mỗi câu được gắn một **danh sách chuỗi đặc trưng** phải **cùng** xuất hiện trong ngữ cảnh truy xuất được. Ví dụ câu 4 yêu cầu đủ cả `"2 tài liệu/1 lần mượn"`, `"20 ngày kể từ ngày nhận"` và `"5.000 đồng/tài liệu/ngày"`. Cả 9 chuỗi đã được `grep` xác nhận tồn tại nguyên văn trong corpus trước khi chốt câu hỏi.

Cách chấm ngây thơ — chỉ kiểm gold `doc_id` có trong top-3 — **thổi phồng kết quả**. Với chiến lược của Tùng, cách đó cho **4/5 câu**; kiểm thêm nội dung thì còn **3/5**. Câu 2 là ví dụ: `luu-hanh-tai-lieu` có ở top-3 (hạng 3, score 0,6929) nhưng chunk được lấy dừng lại **ngay trước** dòng chứa "đạt 25/35 câu" — agent sẽ trả lời "cần tham gia lớp tập huấn" mà không nêu được ngưỡng điểm, tức là thiếu đúng phần định lượng câu hỏi nhắm tới.

### Tổng hợp chất lượng truy xuất của nhóm

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---|---|---|---|
| 1 | Hạn mức mượn *(đánh bẫy)* | Cả 4 đều 2đ | Có, với mọi chiến lược | Chỉ đạt được **nhờ `metadata_filter`**. Không lọc thì cả 4 đều sai. |
| 2 | Điều kiện mượn về nhà | `HeadingChunker` (2đ) | Có với heading; `fixed`/`recursive`/`sentence` lấy đúng tài liệu nhưng chunk cắt trước con số | Câu phân hoá mạnh nhất giữa các chiến lược |
| 3 | Quy trình đặt phòng | `RecursiveChunker` (2đ), `HeadingChunker` (1đ) | Có với recursive; `fixed` và `sentence` mất gold doc khỏi top-3 | `quy-dinh-su-dung-thu-vien` mô tả cùng quy trình bằng từ ngữ gần trùng → retrieval hay lấy nhầm |
| 4 | Mượn liên thư viện | `fixed`/`recursive`/`heading` đều 2đ | Có (trừ `sentence`) | Câu dễ nhất — cả 3 con số nằm liền nhau trong một mục ngắn, score cao nhất cả bộ (**0,8043**) |
| 5 | Lệ phí và thời gian trả thẻ | `fixed`/`recursive`/`sentence` (2đ) | Có; `heading` mất vì tách bảng khỏi mục thời gian | Overlap là yếu tố quyết định |

Kết quả top-1 của từng câu (chiến lược `FixedSizeChunker`, trích từ `ket_qua_benchmark.txt`):

| # | Top-1 chunk | Score | Điểm |
|---|---|---|---|
| 1 | `han-muc-muon-tai-lieu-sinh-vien#0` | +0,6956 | 2đ |
| 2 | `han-muc-muon-tai-lieu-sinh-vien#4` | +0,7299 | 0đ |
| 3 | `quy-dinh-su-dung-thu-vien#20` | +0,6371 | 0đ |
| 4 | `muon-lien-thu-vien#3` | +0,8043 | 2đ |
| 5 | `huong-dan-su-dung-thu-vien#5` | +0,7200 | 2đ |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**

Có, và câu 1 là bằng chứng sạch. A/B chạy trên chiến lược `FixedSizeChunker`:

| | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| **KHÔNG filter** | `muon-lien-thu-vien#3` (0,7505) — *2 tài liệu/20 ngày* | `han-muc-muon-tai-lieu-sinh-vien#0` (0,6956) — *3/10* | `han-muc-muon-tai-lieu-giang-vien#0` (0,6866) — *3/180* |
| **CÓ `audience=student`** | `han-muc-muon-tai-lieu-sinh-vien#0` (0,6956) | `han-muc-muon-tai-lieu-sinh-vien#4` (0,6769) | `huong-dan-su-dung-thu-vien#5` (0,5521) |

Không lọc thì ngữ cảnh gửi cho LLM chứa **ba con số khác nhau cho cùng một câu hỏi**, và con số **sai lại có score cao nhất**. Agent gần như chắc chắn trả lời sai đối tượng, hoặc tệ hơn là trộn lẫn ba mức hạn mức. Bật filter thì tài liệu của giảng viên và liên thư viện bị loại ngay từ vòng ứng viên.

Lưu ý điểm số **giảm** khi bật filter (0,7505 → 0,6956). Đúng như kỳ vọng: filter không làm chunk nào giống câu hỏi hơn, nó chỉ loại ứng viên sai đối tượng để nhường slot. Điểm thấp hơn nhưng câu trả lời đúng — **score cao không đồng nghĩa với retrieval tốt**.

Đánh đổi: filter là bộ lọc cứng. Nếu người hỏi là giảng viên mà hệ thống mặc định lọc `audience=student` thì tài liệu đúng bị loại thẳng. Trong sản phẩm thật, giá trị filter phải đến từ danh tính người đăng nhập chứ không phải hằng số trong code.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**

1. **Cách chấm quyết định kết luận.** Chỉ kiểm gold `doc_id` trong top-3 cho 4/5 câu; kiểm thêm "ngữ cảnh có thật chứa đáp án không" thì còn 3/5. Chênh lệch đó chính là mức thổi phồng của cách chấm ngây thơ.

2. **Metadata filter cứu một câu mà không chiến lược chunking nào cứu được.** Câu 1 không lọc thì cả bốn chiến lược đều đưa con số sai lên top-1. Chất lượng retrieval không chỉ nằm ở chunking.

3. **Chiến lược chunking phải khớp cấu trúc thật của tài liệu.** `SentenceChunker` được 4/10 không phải vì code sai mà vì corpus là bảng và danh sách gạch đầu dòng — gần như không có dấu chấm để bám vào, sinh ra chunk 1.411 ký tự.

4. **Tầng embedding phải kiểm trước tầng chunking.** Nhóm chạy thử bằng `MockEmbedder` trước: cùng chiến lược `fixed` cho **0/10 với mock so với 6/10 với `LocalEmbedder`**. Mọi kết luận về chunking sẽ vô nghĩa nếu tầng dưới là nhiễu.

**Bài học rút ra khi so sánh trong nhóm:**

Cùng corpus, cùng năm câu hỏi, cùng embedding, bốn chiến lược cho 4, 6, 7 và 8 điểm — nhưng điều đáng nói không phải thứ hạng mà là **chúng thắng ở những câu khác nhau**. `HeadingChunker` là chiến lược duy nhất trả lời được câu 2; `RecursiveChunker` là chiến lược duy nhất được trọn điểm câu 3; `HeadingChunker` lại là chiến lược duy nhất **thua** câu 5. Không ai chiếm trọn cả năm.

Suy ra: đọc một con số tổng như "8/10" mà không xem phân bố theo từng câu là bỏ sót toàn bộ thông tin hữu ích. Nếu sản phẩm thật chủ yếu nhận câu hỏi dạng quy trình thì `RecursiveChunker` là lựa chọn đúng; nếu chủ yếu là tra cứu số liệu trong bảng thì overlap quan trọng hơn ranh giới ngữ nghĩa.

**Phân tích lỗi (Failure Analysis) — câu 3**

*Câu hỏi nào hỏng:* Câu 3 — "Quy trình đăng ký và sử dụng phòng học nhóm gồm những bước nào, và đến trễ bao lâu thì kết quả đặt phòng bị hủy?". Với `FixedSizeChunker` và `SentenceChunker`, gold doc `su-dung-phong-hoc-nhom` **vắng mặt hoàn toàn** khỏi top-3. Top-1 là `quy-dinh-su-dung-thu-vien#20` với score 0,6371.

*Tại sao:* hai nguyên nhân chồng lên nhau.

Thứ nhất, `quy-dinh-su-dung-thu-vien` mục 9 mô tả **cùng một quy trình đặt phòng** bằng từ ngữ gần như trùng khớp: "đăng ký trực tiếp tại Quầy thông tin, hoặc đăng ký trực tuyến… chọn mục ĐẶT PHÒNG", "đến trễ trên 15 phút". Cosine đo độ giống chủ đề, và hai tài liệu này giống nhau thật — nên chunk từ tài liệu sai chiếm top-1 là hợp lý về mặt toán học. Đây là hệ quả của việc corpus có **nội dung chồng lấn** giữa văn bản quy định tổng và tài liệu hướng dẫn chi tiết.

Thứ hai, câu hỏi đòi **hai thông tin nằm ở hai mục khác nhau** của tài liệu gold: quy trình 3 bước ở mục "Hướng dẫn sử dụng", mốc 15 phút ở mục "Lưu ý". `FixedSizeChunker` cắt cứng theo 500 ký tự nên xé chúng ra, không chunk nào chứa đủ cả hai.

*Đề xuất cải thiện:* ba hướng, theo thứ tự chi phí tăng dần.

- **Tăng `top_k` từ 3 lên 5** — rẻ nhất, nhưng làm loãng ngữ cảnh và tăng chi phí token.
- **Dùng chunker theo heading có overlap** — giải quyết nguyên nhân thứ hai, đúng như kết luận ở mục 2. `HeadingChunker` hiện đã được 1đ ở câu này, thêm overlap có thể lên 2đ.
- **Thêm trường lọc `category`** — câu hỏi về phòng học nhóm có thể lọc `category=facility`, loại thẳng `quy-dinh-su-dung-thu-vien` (`category=regulation`) khỏi vòng ứng viên. Nhóm đã có sẵn trường này trong metadata nhưng chưa dùng; đây là việc đầu tiên sẽ làm nếu có thêm thời gian.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**

Thứ nhất, **tách file theo `category` chứ không chỉ theo `audience`**. Corpus hiện có nội dung chồng lấn thật: quy định đặt phòng xuất hiện ở cả `quy-dinh-su-dung-thu-vien` lẫn `su-dung-phong-hoc-nhom`; hạn mức mượn xuất hiện ở cả `quy-dinh-su-dung-thu-vien`, `luu-hanh-tai-lieu` và hai file hạn mức. Chồng lấn này gây ra trực tiếp failure case câu 3.

Thứ hai, **chọn nguồn có công bố phiên bản**. Cả 8 tài liệu đều `document_version: not-stated` nên không kiểm thử được kịch bản tài liệu cũ và mới cùng tồn tại — một trong những vấn đề thực tế nhất của hệ thống tra cứu quy định.

Thứ ba, **xử lý mâu thuẫn trong nguồn ngay từ khâu thu thập**. Nhóm phát hiện trang Quy định ghi sinh viên được gia hạn **1 lần** còn trang Lưu hành ghi **2 lần**. Nhóm giữ nguyên cả hai theo đúng nguồn và ghi chú chéo trong file, nhưng phải loại câu hỏi về số lần gia hạn khỏi bộ benchmark. Lần sau nên liên hệ đơn vị chủ quản để xác nhận, hoặc chọn corpus không có mâu thuẫn nội tại.

Thứ tư, **thống nhất embedding backend từ đầu buổi**. Nhóm mất thời gian vì ban đầu mỗi người dùng một backend khác nhau (mock, OpenAI không có key, local). Cài `requirements-local.txt` ngay từ CHECKPOINT 1 để tải model chạy nền trong lúc code là cách đúng.

---

## Cách tái lập toàn bộ số liệu trong báo cáo này

```powershell
pip install -r requirements.txt
pip install -r requirements-local.txt          # LocalEmbedder, không cần API key

pytest tests/ -v                                # 42 passed

$env:PYTHONIOENCODING="utf-8"
python bench.py --all                           # bảng so sánh 4 chiến lược
python bench.py --strategy fixed | Out-File -Encoding utf8 ket_qua_benchmark.txt
```

Lưu ý `PYTHONIOENCODING` và `Out-File -Encoding utf8`: dùng `>` của PowerShell sẽ ghi file UTF-16 và làm hỏng tiếng Việt.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|:----------------:|
| Lựa chọn tài liệu (Document Set Quality) | 9 / 10 |
| Thiết kế chiến lược (Strategy Design) | 13 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 8 / 10 |
| Thuyết trình (Demo) | 4 / 5 |
| **Tổng phần nhóm** | **34 / 40** |

Căn cứ tự chấm:

- **Lựa chọn tài liệu 9/10** — 8 tài liệu đủ metadata, `sources.csv` khớp 1-1, nguồn minh bạch, quy trình crawl tuân thủ `robots.txt` và ghi rõ 6 nguồn đã loại kèm lý do. Trừ 1 điểm vì `document_version` là `not-stated` toàn bộ và hai trường `department`/`language` hiện là hằng số, chưa đóng góp gì cho retrieval.
- **Thiết kế chiến lược 13/15** — bốn chiến lược khác nhau, có baseline đo trên 2 tài liệu, có bảng điểm theo từng câu, và giải thích được vì sao mỗi chiến lược thắng ở câu nào. Trừ 2 điểm vì đề xuất "heading + overlap" mới dừng ở mức ý tưởng, chưa cài đặt và đo thử.
- **Chất lượng truy xuất 8/10** — theo điểm cao nhất trong nhóm (`RecursiveChunker` của Đức Anh, 8/10), chấm bằng thang nghiêm yêu cầu ngữ cảnh chứa đủ mọi chuỗi đặc trưng của gold answer.
- **Thuyết trình 4/5** — đây là **tự đánh giá mức chuẩn bị**, ghi trước buổi demo. Nhóm đã có đủ ba thứ lab doc yêu cầu trình bày: chiến lược riêng của từng thành viên kèm lý do, bảng so sánh trong nhóm với điểm theo từng câu, và bài học rút ra. `bench.py` chạy được bằng một lệnh nên demo trực tiếp không phải debug tại chỗ. Trừ 1 điểm vì nhóm chưa tập luyện phần chia lượt nói giữa bốn người trong 6–8 phút, và chưa chuẩn bị câu trả lời cho câu hỏi "nhóm học được gì từ nhóm khác" — phần này chỉ trả lời được sau khi nghe các nhóm còn lại.
