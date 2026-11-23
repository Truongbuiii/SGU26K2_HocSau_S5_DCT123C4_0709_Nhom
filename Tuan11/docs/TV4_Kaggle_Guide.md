# Hướng dẫn hoàn tất TV4 trên Kaggle

**Người phụ trách:** Lê Thị Thanh Tuyền – 3123411332.
**Ngày cập nhật hướng dẫn:** 24/11/2026.

Các timestamp do code mới ghi vào `env.json` và nhật ký sẽ theo múi giờ UTC+7 (giờ Việt Nam/Thái Lan), không theo đồng hồ UTC của máy Kaggle.

Theo phân công, TV4 phụ trách trực quan hóa filter, E4 pretrained, chạy mô hình tốt nhất với nhiều seed, phân tích, report và slide. Code đã có trong `src/`; hướng dẫn này chỉ rõ những phần phải chạy để tạo kết quả thật.

## Trạng thái repo hiện tại

- Bảng `results/summary.csv` có baseline và kết quả E2/E3. Ứng viên hiện có điểm validation cao nhất là **E2-04** (Medium, Adam, lr=1e-4; val accuracy 66.397%). Cần TV2/TV3 xác nhận đã tổng hợp đủ E1–E3 trước khi nhóm chốt cấu hình. Chỉ chọn theo validation, không chọn theo cột test.
- Các thư mục run đã commit không có `best.pt`. Chỉ có log, lịch sử và hình đánh giá; không thể dựng feature map/activation từ các file này. Cần chạy lại cấu hình CNN cuối trên Kaggle để giữ checkpoint.
- FER2013 CSV không nằm trong repo. Cần attach CSV riêng trong Kaggle.
- Run E4, ảnh minh chứng Kaggle, nhật ký TV4 và kết quả test cuối chưa có.

## 1. Tạo Kaggle Notebook

1. Tạo Notebook mới, bật **Internet** để tải ImageNet pretrained weights, chọn **GPU T4** (hoặc GPU khả dụng khác).
2. Chọn **Add Input** và thêm FER2013 CSV của nhóm. Kiểm tra đúng file `fer2013.csv`, có cột `emotion`, `pixels`, `Usage` và các split `Training`, `PublicTest`, `PrivateTest` lần lượt 28,709 / 3,589 / 3,589.
3. Trong cell đầu, clone repo:

```python
!git clone https://github.com/Truongbuiii/SGU26K2_HocSau_S5_DCT123C4_0709_Nhom.git /kaggle/working/repo
%cd /kaggle/working/repo/Tuan11
```

4. Tìm file CSV:

```python
!find /kaggle/input -name fer2013.csv
```

Đặt biến bằng đường dẫn thực tế được in ra:

```python
CSV = "/kaggle/input/<dataset-slug>/fer2013.csv"
!nvidia-smi
!FER2013_CSV="$CSV" python src/dataset.py
```

Nếu Kaggle báo thiếu package, cài package còn thiếu. Không cài đè PyTorch/CUDA đang có sẵn nếu import và GPU đã hoạt động.

## 2. Chạy E4-01 ResNet18 và E4-02 VGG16

```python
!python src/train_pretrained.py --arch resnet18 --seed 42 --csv_path "$CSV" --output_root /kaggle/working/results/runs
!python src/train_pretrained.py --arch vgg16 --seed 42 --csv_path "$CSV" --output_root /kaggle/working/results/runs
```

Mỗi run chọn checkpoint theo `val_loss` và ghi `config.json`, `env.json`, `history.csv`, `best_metrics.json`, `best.pt`. Script bật AMP trên GPU; giữ cùng thiết lập cho cả hai thí nghiệm. VGG16 nặng hơn. Nếu session ngắt, lưu artifacts đang có và ghi run lỗi/dở vào nhật ký trước khi quyết định chạy lại.

## 3. Chạy cấu hình tốt nhất với seed 42, 43, 44

Trước khi chạy, nhóm chốt cấu hình theo bảng đầy đủ E1–E3. Theo summary hiện tại, E2-04 là ứng viên: Medium, ReLU, không BN, conv dropout 0.2, FC dropout 0.5, `flip_affine`, `div255`, Adam `lr=1e-4`, batch 64, tối đa 100 epoch, patience 10. Nếu TV2/TV3 xác nhận cấu hình này, chạy:

```python
!FER2013_CSV="$CSV" python src/train.py --exp_id E4-03 --exp_name best_seed42 --runner TV4 --arch medium --activation relu --conv_dropout 0.2 --fc_dropout 0.5 --optimizer adam --lr 1e-4 --batch_size 64 --aug_name flip_affine --norm div255 --epochs 100 --patience 10 --seed 42
!FER2013_CSV="$CSV" python src/train.py --exp_id E4-03 --exp_name best_seed43 --runner TV4 --arch medium --activation relu --conv_dropout 0.2 --fc_dropout 0.5 --optimizer adam --lr 1e-4 --batch_size 64 --aug_name flip_affine --norm div255 --epochs 100 --patience 10 --seed 43
!FER2013_CSV="$CSV" python src/train.py --exp_id E4-03 --exp_name best_seed44 --runner TV4 --arch medium --activation relu --conv_dropout 0.2 --fc_dropout 0.5 --optimizer adam --lr 1e-4 --batch_size 64 --aug_name flip_affine --norm div255 --epochs 100 --patience 10 --seed 44
```

Các cấu hình E1–E3 hiện có ghi `use_amp=false`, nên ví dụ cũng tắt AMP để giữ thiết lập nhất quán. Nếu nhóm đã chốt giao thức AMP khác, dùng thống nhất trên cả ba seed và ghi vào nhật ký. Nếu nhóm chọn cấu hình khác, thay tham số đúng theo cấu hình đó. Ghi val accuracy tốt nhất của ba seed và tính mean ± std.

Tự tổng hợp best epoch và mean ± độ lệch chuẩn mẫu bằng script:

```python
!python src/summarize_tv4.py --run_dirs results/runs/E4-03_best_seed42_s42 results/runs/E4-03_best_seed43_s43 results/runs/E4-03_best_seed44_s44 --out results/summary_TV4_seeds.csv
```

## 4. Tạo hình filter cho checkpoint CNN cuối

Ví dụ với seed 42 (đường dẫn run cần khớp tên thực tế mà `train.py` in ra):

```python
!python src/visualize_filters.py --run_dir results/runs/E4-03_best_seed42_s42 --csv_path "$CSV"
!python src/gradcam.py --run_dir results/runs/E4-03_best_seed42_s42 --csv_path "$CSV" --out results/figures/gradcam_final.png
```

`visualize_filters.py` sinh ba hình: kernel conv đầu, feature map lớp conv cuối với ví dụ các lớp cảm xúc, và activation maximization cho một số filter cuối. Grad-CAM là tùy chọn. Lưu các PNG trong report/slide.

## 5. Đánh giá PrivateTest sau khi khóa mô hình

Chỉ đánh giá sau khi cấu hình và epoch đã được chọn bằng validation. Với pretrained:

```python
!python src/evaluate_pretrained.py --run_dir /kaggle/working/results/runs/E4-resnet18_resnet18_s42 --csv_path "$CSV"
```

Thay `run_dir` bằng checkpoint pretrained cuối cần báo cáo. Với mô hình CNN nội bộ, nếu `evaluate.py` của TV3 chưa được thêm vào repo, dùng fallback TV4:

```python
!python src/evaluate_final_cnn.py --run_dir results/runs/E4-03_best_seed42_s42 --csv_path "$CSV"
```

Chỉ chạy với checkpoint cuối đã khóa và lưu kết quả; không dùng PrivateTest để đổi lựa chọn mô hình.

## 6. Lưu minh chứng và hoàn thành báo cáo/slide

- Sau khi chạy, dùng **Save Version → Save & Run All** và giữ URL/version Kaggle. Đảm bảo thư mục `/kaggle/working/results` được lưu trong output notebook; tải xuống hoặc lưu thành Kaggle Dataset nếu cần giữ lâu dài.
- Chụp minh chứng có tên GPU, cell lệnh, log epoch, kết quả val cuối và thời gian; lưu dưới `Tuan11/evidence/ThanhVien4/`.
- Ghi ngay nhật ký: ExpID, ngày giờ, người chạy, mục tiêu, thay đổi baseline, seed, best epoch, val loss/accuracy, test metrics (chỉ sau khi khóa), thời gian, đường dẫn run, nhận xét.
- Hoàn thiện `docs/TV4_Report_Slide_Draft.md` bằng số liệu thật, hình thật, nhận xét train/val gap, lớp dễ nhầm, so sánh ResNet18/VGG16 và mean ± std của ba seed.
- Commit/push code, nhật ký, report draft đã hoàn thiện, hình và đường dẫn Kaggle Version theo quy ước nhánh của nhóm. Không commit FER2013 CSV hoặc checkpoint lớn vào Git.
