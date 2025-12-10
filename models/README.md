# Models Directory

Models không được upload lên GitHub do kích thước lớn.

## Download Pre-trained Models

### Option 1: Tự train model
```bash
python scripts/train_model.py
```

### Option 2: Download từ external hosting

#### Random Forest Classifier (trained on 800+ images)
- File: `random_forest_classifier.pkl` (~10-50MB)
- Download: [Link sẽ được cập nhật sau khi upload]
- Accuracy: ~79% (EXIF only) hoặc ~88% (EXIF + Face)

#### YOLOv8 Face Detection Model
- File: `yolov8n-face.pt` (~6MB)
- Download: Tự động download khi chạy lần đầu
- Hoặc: https://github.com/akanametov/yolov8-face/releases

### Installation
```bash
# Đặt models vào thư mục này:
models/
├── random_forest_classifier.pkl
└── yolov8n-face.pt
```

## Model Info

### Random Forest Classifier
- Input: 10 EXIF features (hoặc 13 nếu có face detection)
- Output: 5 categories (ChanDung, TinhVat, TheThao, PhongCanh, DongVat)
- Training: 500 trees, max_depth=15, balanced weights

### YOLOv8 Face
- Input: RGB images
- Output: Face bounding boxes + confidence scores
- Purpose: Detect faces để phân biệt ChanDung vs TinhVat
