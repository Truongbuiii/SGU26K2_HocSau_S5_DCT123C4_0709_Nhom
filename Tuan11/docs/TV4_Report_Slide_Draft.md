# Phần TV4 – Trực quan hóa, E4 và phân tích

**Thành viên phụ trách:** Lê Thị Thanh Tuyền – 3123411332  
**Trạng thái:** Bản khung; hoàn tất các ô `[điền sau khi chạy Kaggle]` bằng số liệu và hình thực tế. Không điền kết quả dự đoán trước.

## Nội dung báo cáo

### 1. Trực quan hóa filter và biểu diễn học được

Mô hình CNN cuối được chọn theo độ chính xác/loss trên PublicTest. Trực quan hóa kernel lớp convolution đầu giúp quan sát các mẫu cạnh, tương phản và cấu trúc cục bộ mà mạng học. Activation lớp convolution cuối được trình bày theo ảnh mẫu từng cảm xúc để thấy đáp ứng không gian của các kênh sâu. Activation maximization tối ưu ảnh đầu vào để kích hoạt một số filter cuối; đây là hình diễn giải định tính, không phải ảnh khuôn mặt thực hay bằng chứng duy nhất cho quyết định của mô hình.

**Mô hình/checkpoint:** `[điền ExpID, seed, best epoch]`  
**Hình kernel đầu:** `results/figures/[tên hình]`  
**Hình feature map cuối:** `results/figures/[tên hình]`  
**Hình activation maximization:** `results/figures/[tên hình]`  
**Nhận xét quan sát:** `[điền sau khi xem các hình; nêu điểm chung/khác theo lớp, tránh suy diễn quá mức]`

### 2. E4: fine-tune pretrained

FER2013 là ảnh xám 48×48, trong khi torchvision ImageNet models nhận ba kênh. Pipeline nhân bản kênh xám thành RGB, resize lên 96×96, rồi chuẩn hóa theo mean/std ImageNet. Thay lớp phân loại cuối để dự đoán bảy cảm xúc; fine-tune toàn bộ model bằng AdamW. Checkpoint được chọn theo val loss trên PublicTest.

| Run | Mô hình | Seed | Best epoch | Val loss | Val accuracy | Thời gian | Ghi chú |
|---|---|---:|---:|---:|---:|---:|---|
| E4-01 | ResNet18 pretrained | 42 | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| E4-02 | VGG16 pretrained | 42 | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Phân tích sau khi có kết quả:** so sánh validation, tốc độ, train/validation gap và mức độ ổn định. Không mặc định pretrained tốt hơn hoặc kém hơn CNN tự xây. `[điền nhận xét dựa trên log thực tế]`

### 3. Mô hình tốt nhất qua nhiều seed

Mô hình được chọn theo validation sau khi tổng hợp E1–E3. Các seed dùng cùng cấu hình và thiết lập AMP.

| Seed | Best epoch | Val accuracy | Val loss | PrivateTest accuracy* |
|---:|---:|---:|---:|---:|
| 42 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 43 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 44 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| **Mean ± std** | — | `[ ]` | `[ ]` | `[ ]` |

\* PrivateTest chỉ được đánh giá sau khi đã khóa cấu hình theo PublicTest. Báo cáo riêng test cuối; không dùng test để chọn model hoặc epoch.

### 4. Phân tích và kết luận TV4

- Yếu tố quan trọng nhất theo tổng hợp E1–E3: `[điền sau khi TV3 tổng hợp]`.
- Fine-tune ResNet18/VGG16 so với CNN tự xây: `[điền theo validation và chi phí chạy]`.
- Độ biến động qua seed: `[mean ± std, diễn giải thận trọng]`.
- Lớp có recall thấp/nhầm nhiều và lý do khả dĩ: `[tham khảo confusion matrix/per-class metrics]`.
- Hạn chế: dữ liệu mất cân bằng, ảnh nhỏ/xám, nhãn cảm xúc chủ quan, validation dao động, chi phí fine-tune: `[chọn điều đúng với kết quả]`.
- Kết luận: `[viết sau khi đã có mọi kết quả; không khẳng định vượt mốc nếu số liệu không hỗ trợ]`.

## Dàn ý slide TV4

1. **Filter và feature map:** CNN cuối/checkpoint; kernel lớp đầu và feature maps lớp cuối; chú thích lớp ảnh minh họa.
2. **Activation maximization/Grad-CAM:** một số kênh cuối và/hoặc ca đúng/sai; nêu đây là diễn giải định tính.
3. **E4 transfer learning:** bảng ResNet18/VGG16, quy trình RGB 96×96/ImageNet normalization, validation và thời gian.
4. **Đa seed và kết luận:** cấu hình được chọn theo validation, mean ± std seed 42/43/44, kết luận và hạn chế.

## Kết quả hiện có trong repo (để đối chiếu, không thay cho kết luận cuối)

`results/summary.csv` hiện ghi E2-04 có val accuracy 66.397%; đây là ứng viên tốt nhất trong các dòng summary hiện có. E2-03 (Adam lr=1e-2) đạt val accuracy 24.937%, có thể làm fail case thực tế nếu kết quả này còn hợp lệ sau kiểm tra chung. Chỉ dùng những kết quả sau khi TV3 xác nhận bảng đủ và nhật ký khớp; không chọn dựa trên cột test.
