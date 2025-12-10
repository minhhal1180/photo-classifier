# Photo Classifier - Quick Start Guide

##  Cài đặt nhanh

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu trúc thư mục
```bash
python scripts/setup_project.py
```

### 3. Kiểm tra dataset
```bash
python scripts/verify_dataset.py
```

##  Sử dụng

### Training (Huấn luyện model)

```bash
# Huấn luyện với tất cả features
python src/main.py train

# Huấn luyện không dùng face detection (nhanh hơn)
python src/main.py train --no-face-detect

# Sử dụng EXIF data có sẵn (không extract lại)
python src/main.py train --skip-exif
```

### Inference (Phân loại ảnh mới)

```bash
# Phân loại ảnh cơ bản
python src/main.py classify

# Phân loại với face recognition
python src/main.py classify --recognize

# Tổ chức theo người (cần --recognize)
python src/main.py classify --recognize --organize-person

# Di chuyển file thay vì copy
python src/main.py classify --move
```

### Xem cấu hình

```bash
python src/main.py info
```

##  Cấu trúc Dataset

### Training Images
```
data/training_images/
├── ChanDung/       # Ảnh chân dung
├── TinhVat/        # Ảnh tĩnh vật
├── TheThao/        # Ảnh thể thao
├── PhongCanh/      # Ảnh phong cảnh
└── DongVat/        # Ảnh động vật
```

### Inference Images
```
data/inference_images/
└── [Đặt ảnh cần phân loại vào đây]
```

### Known Faces (cho face recognition)
```
data/known_faces/
├── NguyenVanA/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── TranThiB/
    └── photo1.jpg
```

##  Cấu hình

Chỉnh sửa `config.yaml` để thay đổi:
- Đường dẫn thư mục
- Tham số model (số cây, depth, etc.)
- Ngưỡng face detection/recognition
- Device (cuda/cpu)

##  Output

### Training
```
outputs/training_results/YYYYMMDD_HHMMSS/
├── classification_report.txt    # Chi tiết độ chính xác
├── confusion_matrix.png         # Ma trận nhầm lẫn
└── feature_importance.png       # Mức độ quan trọng của features
```

### Inference
```
outputs/classified_images/YYYYMMDD_HHMMSS/
├── by_category/                 # Ảnh được tổ chức theo danh mục
│   ├── ChanDung/
│   ├── TinhVat/
│   └── ...
├── by_person/                   # Ảnh được tổ chức theo người (nếu bật)
│   ├── NguyenVanA/
│   └── TranThiB/
├── classification_results.csv   # Kết quả chi tiết (CSV)
└── summary_report.txt          # Báo cáo tổng hợp
```


