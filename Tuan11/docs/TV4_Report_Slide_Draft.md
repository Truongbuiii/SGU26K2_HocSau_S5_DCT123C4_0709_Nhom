# Phần TV4 – Trực quan hóa, E4 và phân tích

**Thành viên phụ trách:** Lê Thị Thanh Tuyền – 3123411332  
**Ngày chạy theo minh chứng:** 24/11/2026 (UTC+7)  
**Trạng thái:** Đã điền kết quả đọc từ minh chứng Kaggle. Bổ sung các ảnh PNG gốc nếu tải được từ Kaggle Output; bản DOCX minh chứng và ảnh chụp màn hình đã lưu dưới `evidence/ThanhVien4/`.

## 1. Trực quan hóa filter và biểu diễn học được

Checkpoint CNN được trực quan hóa là **E4-03, seed 42**, cấu hình Medium theo E2-04, epoch tốt nhất 43. Kernel convolution đầu cho thấy các bộ lọc tương phản/cạnh ở mức ảnh xám. Feature map lớp convolution cuối được minh họa trên ảnh validation thuộc các lớp cảm xúc; các kênh phản ứng khác nhau theo vùng mặt. Activation maximization tạo mẫu đầu vào tổng hợp để kích hoạt các filter 0, 16, 64 và 128. Đây là cách diễn giải định tính, không phải ảnh khuôn mặt thực hay bằng chứng độc lập cho quyết định của mô hình.

Các hình trong minh chứng Kaggle:

- `results/figures/E4-03_best_seed42_s42_first_kernels.png`
- `results/figures/E4-03_best_seed42_s42_last_feature_maps.png`
- `results/figures/E4-03_best_seed42_s42_activation_maximization.png`
- Tùy chọn: `results/figures/gradcam_final.png`

**Nhận xét:** feature maps cho thấy vùng mắt, mũi và miệng có đáp ứng rõ; Grad-CAM tập trung nhiều vào vùng mặt, đặc biệt quanh mắt/miệng. Các hình là ví dụ định tính; chúng không chứng minh mạng luôn dùng đúng vùng trên mọi ảnh.

## 2. E4: fine-tune pretrained

FER2013 ảnh xám 48×48 được nhân bản thành ba kênh, resize lên 96×96 và chuẩn hóa theo ImageNet. Thay lớp phân loại cuối để dự đoán bảy cảm xúc; fine-tune với AdamW và chọn checkpoint theo val loss trên PublicTest.

| Run | Mô hình | Seed | Best epoch | Val loss | Val accuracy | PrivateTest accuracy | Thời gian |
|---|---|---:|---:|---:|---:|---:|---:|
| E4-01 | ResNet18 pretrained | 42 | 7 | 0.9412 | 65.90% | 67.43% | 190.98 s |
| E4-02 | VGG16 pretrained | 42 | 11 | 0.9241 | 68.32% | 69.27% | 1339.24 s |

Trong hai cấu hình pretrained đã chạy, VGG16 cao hơn ResNet18 **2.42 điểm phần trăm trên validation**. VGG16 cần thời gian lâu hơn (khoảng 22.3 phút so với 3.2 phút). Learning curves cho thấy train accuracy tiếp tục tăng sau khi validation chững lại, rõ hơn ở VGG16; early stopping theo validation vì vậy hữu ích. PrivateTest được báo cáo sau khi khóa lựa chọn, không dùng để chọn cấu hình.

## 3. Cấu hình CNN tốt nhất qua ba seed

Theo kết quả E1–E3 của nhóm, E2-04 được dùng làm cấu hình E4-03: Medium, ReLU, không BatchNorm, conv dropout 0.2, FC dropout 0.5, `flip_affine`, `div255`, Adam `lr=1e-4`, batch 64, tối đa 100 epoch, patience 10. AMP tắt, đồng nhất với config các run hiện có.

| Seed | Best epoch | Val loss | Val accuracy | PrivateTest accuracy |
|---:|---:|---:|---:|---:|
| 42 | 43 | 0.939398 | 66.3973% | 67.29% |
| 43 | 57 | 0.931446 | 66.8153% | 67.51% |
| 44 | 43 | 0.928837 | 66.4531% | 68.01% |
| **Mean ± sample std** | — | **0.9332 ± 0.0055** | **66.56 ± 0.23%** | **67.60 ± 0.37%** |

Độ lệch chuẩn validation 0.23 điểm phần trăm và PrivateTest 0.37 điểm phần trăm cho thấy các run đa seed khá ổn định trong lần chạy này. Không chọn seed theo PrivateTest; báo cáo test trung bình chỉ để mô tả hiệu năng sau khi cấu hình đã khóa.

## 4. Phân tích và kết luận TV4

- **Transfer learning:** VGG16 là cấu hình cao nhất trong hai run pretrained theo validation (68.32%). Kết quả này cao hơn ResNet18 (65.90%) và E4-03 CNN trung bình (66.56%). Đổi lại, VGG16 tốn thời gian fine-tune nhiều hơn và có khoảng cách train/validation rõ hơn.
- **Độ ổn định:** E4-03 ba seed đạt 66.56 ± 0.23% validation; test trung bình 67.60 ± 0.37%. Chênh lệch nhỏ giữa các seed cần được diễn giải thận trọng.
- **Lỗi theo lớp:** confusion matrix và per-class report cho thấy lớp Disgust có support thấp (55 ảnh trong PrivateTest), nên recall biến động mạnh; các nhầm lẫn giữa Fear/Sad/Neutral cũng đáng chú ý. Dùng confusion matrix/per-class metrics trong `screenshots/14_PrivateTest_CNN.png` để trích số chính xác khi viết bản báo cáo cuối.
- **Fail case:** E2-03 (Adam `lr=1e-2`) trong summary nhóm đạt 24.937% validation; có thể dùng làm kết quả không đạt mong đợi nếu TV3 xác nhận bảng tổng hợp.
- **Kết luận:** transfer learning với VGG16 cho validation cao nhất trong các run E4, còn CNN được chọn từ E1–E3 đạt kết quả nhất quán qua ba seed. Kết quả không nên so trực tiếp với mốc bên ngoài nếu khác preprocessing, split hoặc quy trình đánh giá.

## 5. Dàn ý slide TV4

1. **Filter và feature map:** ba hình kernel, activation map và activation maximization; ghi rõ CNN seed 42, best epoch 43.
2. **Grad-CAM:** minh họa một số ảnh đúng/sai từ PublicTest; diễn giải vùng mạng chú ý là định tính.
3. **E4 pretrained:** bảng ResNet18/VGG16 với val accuracy, best epoch và runtime.
4. **Đa seed và kết luận:** cấu hình E2-04, mean ± std ba seed, PrivateTest sau khi khóa mô hình, hạn chế và fail case E2-03.

## Nguồn số liệu và minh chứng

Số liệu trên được chép từ notebook Kaggle và 23 ảnh trong `evidence/ThanhVien4/MinhChung_TV4_Kaggle_2026-11-24.docx`. Dòng thời gian gốc trong notebook hiện ảnh có thể dùng UTC ngày 23; ngày chạy địa phương là 24/11/2026 UTC+7. Hãy dùng timestamp trong các `env.json` gốc nếu cần ghi chính xác giờ bắt đầu/kết thúc từng run.
