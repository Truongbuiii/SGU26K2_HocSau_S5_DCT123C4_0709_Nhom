# Hướng dẫn TV4: E4, trực quan hóa và phần báo cáo

Tài liệu phân công yêu cầu TV4 chạy E4 gồm fine-tune ResNet18, VGG16 và mô hình tốt nhất với seed 42/43/44; chọn mô hình theo PublicTest (validation), chỉ dùng PrivateTest để báo cáo kết quả cuối. Không kết luận trước fine-tune thất bại. Người chạy tự giữ Kaggle Notebook Version và ảnh minh chứng.

## 1. Chuẩn bị Kaggle Notebook

1. Tạo Notebook mới, chọn **Accelerator = GPU T4** (P100 cũng được). Bật **Internet = On** để lần đầu tải ImageNet pretrained weights.
2. Chọn **Add Input** và thêm FER2013 CSV dataset của nhóm. File phải có `emotion`, `pixels`, `Usage`, với 28,709 Training / 3,589 PublicTest / 3,589 PrivateTest.
3. Đưa thư mục `Tuan11` lên Kaggle bằng một trong hai cách: add repo dạng Kaggle Dataset (khuyến nghị, dữ liệu code riêng với CSV), hoặc clone repo GitHub. Nếu clone:

```python
!git clone <URL_REPO_CUA_NHOM> /kaggle/working/repo
%cd /kaggle/working/repo/Tuan11
```

4. Tìm đường dẫn CSV đã attach. Kaggle thường đặt input trong `/kaggle/input/<dataset-slug>/fer2013.csv`. Xem đường dẫn:

```python
!find /kaggle/input -name fer2013.csv
```

## 2. Chạy hai thí nghiệm pretrained

Chạy từ `Tuan11/`. Thay đường dẫn CSV bằng kết quả `find`:

```python
CSV = "/kaggle/input/fer2013/fer2013.csv"
!python src/train_pretrained.py --arch resnet18 --seed 42 --csv_path "$CSV" --output_root /kaggle/working/results/runs
!python src/train_pretrained.py --arch vgg16 --seed 42 --csv_path "$CSV" --output_root /kaggle/working/results/runs
```

Mỗi run ghi `config.json`, `env.json`, `history.csv`, `best_metrics.json`, và `best.pt`. VGG16 nặng hơn nhiều; nếu Kaggle hết giờ thì giữ nguyên run dở, ghi rõ vào nhật ký, rồi chạy lại với cùng cấu hình. Không tự ý so sánh giữa các run có AMP khác nhau; mặc định script bật AMP trên GPU.

Sau khi TV2/TV3 tổng hợp E1–E3 và nhóm họp chọn cấu hình tốt nhất theo **val**, chạy đúng cấu hình CNN đó cho seed 42, 43, 44 theo `train.py` / giao diện `train(cfg)` mà TV3 cung cấp. Tính trung bình và độ lệch chuẩn của val accuracy. Run E4-01, E4-02 và ba seed cuối tạo tối thiểu năm run E4.

## 3. Tạo hình trực quan hóa CNN cuối

Chọn thư mục checkpoint của mô hình CNN cuối (ví dụ `results/runs/E3-06_meanstd_s42`). Script đọc `config.json` và `best.pt`, rồi lưu ba PNG vào `results/figures/`:

```python
!python src/visualize_filters.py --run_dir results/runs/E3-06_meanstd_s42 --csv_path "$CSV"
!python src/gradcam.py --run_dir results/runs/E3-06_meanstd_s42 --csv_path "$CSV" --out results/figures/gradcam_final.png
```

Ba hình bắt buộc từ `visualize_filters.py`: kernel conv đầu, feature map conv cuối trên một ảnh mỗi lớp, và activation maximization cho một số channel conv cuối. Grad-CAM là phần tùy chọn; hình đang dùng ảnh PublicTest và nhãn dự đoán.

## 4. Minh chứng, kết quả và báo cáo

- Sau mỗi nhóm chạy, tải `results/runs/` và `results/figures/` về máy hoặc lưu vào Kaggle Output bằng **Save Version → Save & Run All**. Giữ URL/version notebook và chụp màn hình GPU, log epoch, kết quả cuối có thời gian.
- Chỉ sau khi đã khóa mô hình và epoch theo validation, chấm PrivateTest đúng một lần cho từng mô hình cuối. Với E4 pretrained:

```python
!python src/evaluate_pretrained.py --run_dir results/runs/E4-resnet18_resnet18_s42 --csv_path "$CSV"
```

Thay run_dir bằng checkpoint cuối đã chọn. Lưu `private_test_metrics.json` và confusion matrix trong run đó.
- Ghi nhật ký ngay: ExpID, người chạy, ngày giờ, thay đổi so với baseline, seed, best epoch, val accuracy/loss, test accuracy sau khi khóa, thời gian, đường dẫn artifact, nhận xét.
- Khi viết báo cáo: giải thích ImageNet pretrained buộc ảnh xám thành 3 kênh, resize 96×96, chuẩn hóa ImageNet; so sánh ResNet18/VGG16 theo validation và nêu train/val gap. Ghi mean ± std cho ba seed. Nếu chênh lệch nhỏ, coi là nhiễu. Không chọn mô hình theo PrivateTest.
- Đưa hình filter/feature map/activation maximization, bảng kết quả E4, nhận xét lớp nhầm nhiều, và kết luận vào phần TV4; ghép bìa, report, slide theo yêu cầu nhóm. Thay header tác giả trong các file TV4 bằng họ tên/MSSV thật trước khi nộp.

## 5. Hạn chế hiện tại cần biết

`train_pretrained.py` lưu checkpoint tốt nhất theo val loss và không chạy PrivateTest. `evaluate_pretrained.py` dành riêng cho checkpoint pretrained. Các script CNN TV4 dùng API `build_model` / `get_last_conv` và định dạng checkpoint hiện có trong repo.
