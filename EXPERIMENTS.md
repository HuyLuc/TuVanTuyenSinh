# Nhật ký thử nghiệm

Mỗi thử nghiệm ghi: ngày, thay đổi, cấu hình (mô hình, tham số), kết quả theo từng nhóm câu hỏi, nhận xét.

## Thông tin môi trường

| Mục | Giá trị | Ngày kiểm tra |
|---|---|---|
| Mô hình trả lời | `gemini-2.5-flash` | 2026-09-25 |
| Mô hình định tuyến | `gemini-2.5-flash-lite` | 2026-09-25 |
| Bản mới hơn có trên key | `gemini-3.8-flash` (lúc thử bị 503 quá tải), `gemini-3.5-flash-lite` (gọi được) | 2026-09-25 |
| Hạn mức gói miễn phí (RPM / RPD / TPM) | cần tra trên Google AI Studio | |

## Bảng so sánh phiên bản

| Phiên bản | Quy chế | Một trường | Số liệu | Nhiều lượt | Ngoài phạm vi | Định tuyến | Ghi chú |
|---|---|---|---|---|---|---|---|
| baseline (RAG, `--include-tables-in-vector`) | | | | | | | |

## Nhật ký

### 2026-09-25: Giai đoạn 0

- Dựng khung dự án: uv, Docker (một service), FastAPI + Gradio, Qdrant local.
- Hàm gọi Gemini dùng chung: thử lại khi gặp 429/5xx, cache SQLite theo mã băm (mô hình, nội dung, cấu hình gồm prompt, tham số, danh sách hàm, JSON schema), gọi hàm, streaming.
- Kiểm tra với API thật (`scripts/smoke_test.py`): gặp 503 thật và tự thử lại thành công; lần gọi lặp lại lấy từ cache (13,3 giây xuống 0,00 giây); bộ định tuyến trả JSON đúng schema và giữ nguyên văn tên trường ("Bách khoa").
- Gọi hàm: ở chế độ AUTO, `gemini-2.5-flash` trả lời bằng chữ thay vì gọi hàm. Phải dùng `force_tool=True` (mode ANY) thì mới gọi `tra_diem_chuan` với tham số đúng. Cần lưu ý khi thiết kế nhánh số liệu ở Giai đoạn 6.
- Việc cần làm: so sánh `gemini-2.5-*` với `gemini-3.8-flash` / `gemini-3.5-flash-lite` khi đã có bộ đánh giá; tra hạn mức gói miễn phí.
