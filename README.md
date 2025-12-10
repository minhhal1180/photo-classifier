#  Hệ thống Phân loại Ảnh Tự động (AI Photo Classifier)

Hệ thống phân loại ảnh thông minh sử dụng Machine Learning (Random Forest) phân loại với exif file của ảnh raw kết hợp Computer Vision (YOLOv8) để tự động phân loại và gán nhãn ảnh theo danh mục và tên người.

##  Tính năng chính

### 1. Phân loại ảnh theo 5 danh mục
- **Chân Dung (Portrait)**: Ảnh có khuôn mặt người
- **Tĩnh Vật (Still Life)**: Ảnh macro, sản phẩm, đồ vật
- **Thể Thao (Sports)**: Ảnh thể thao, chuyển động nhanh
- **Phong Cảnh (Landscape)**: Ảnh thiên nhiên, kiến trúc
- **Động Vật (Wildlife)**: Ảnh động vật hoang dã

### 2. Phát hiện và nhận dạng khuôn mặt
- Phát hiện khuôn mặt bằng YOLOv8-face
- Nhận dạng và gán tên người bằng DeepFace
- Tự động tạo album theo tên người

### 3. Phân loại theo thời gian
- **Sáng**: 5:00 - 12:00
- **Chiều**: 12:00 - 18:00
- **Tối**: 18:00 - 5:00

##  Cấu trúc dự án

```
photo_classifier_project/
├── README.md                          # Tài liệu dự án
├── requirements.txt                   # Dependencies Python
├── config.yaml                        # Cấu hình dự án
├── .gitignore                         # Git ignore
│
├── src/                               # Source code chính
│   ├── __init__.py
│   ├── main.py                        # Entry point chính
│   ├── config.py                      # Load cấu hình
│   │
│   ├── data_processing/               # Xử lý dữ liệu EXIF
│   │   ├── __init__.py
│   │   ├── exif_extractor.py         # Trích xuất EXIF từ ảnh
│   │   └── feature_engineering.py    # Tạo features từ EXIF
│   │
│   ├── face_detection/                # Phát hiện và nhận dạng khuôn mặt
│   │   ├── __init__.py
│   │   ├── yolo_face_detector.py     # YOLOv8 face detection
│   │   └── face_recognizer.py        # DeepFace recognition
│   │
│   ├── models/                        # ML models
│   │   ├── __init__.py
│   │   ├── trainer.py                # Train Random Forest
│   │   └── predictor.py              # Dự đoán danh mục
│   │
│   ├── classification/                # Logic phân loại
│   │   ├── __init__.py
│   │   ├── category_classifier.py    # Phân loại danh mục
│   │   └── portrait_stilllife.py     # Phân biệt chân dung/tĩnh vật
│   │
│   ├── utils/                         # Tiện ích
│   │   ├── __init__.py
│   │   ├── file_utils.py             # Xử lý file/folder
│   │   ├── visualization.py          # Vẽ biểu đồ
│   │   └── logger.py                 # Logging
│   │
│   └── workflows/                     # Quy trình hoàn chỉnh
│       ├── __init__.py
│       ├── training_workflow.py      # Quy trình huấn luyện
│       └── inference_workflow.py     # Quy trình phân loại ảnh mới
│
├── models/                            # Lưu models đã train
│   ├── random_forest_classifier.pkl
│   ├── yolov8n-face.pt               # YOLOv8 face model
│   └── feature_scaler.pkl
│
├── data/                              # Dữ liệu
│   ├── training_images/              # Ảnh huấn luyện (E:\imagee)
│   │   ├── ChanDung/
│   │   ├── TinhVat/
│   │   ├── TheThao/
│   │   ├── PhongCanh/
│   │   └── DongVat/
│   │
│   ├── inference_images/             # Ảnh cần phân loại
│   │
│   ├── known_faces/                  # Database khuôn mặt để nhận dạng
│   │   ├── NguyenVanA/
│   │   │   ├── photo1.jpg
│   │   │   └── photo2.jpg
│   │   ├── TranThiB/
│   │   │   └── photo1.jpg
│   │   └── ...
│   │
│   └── metadata/                     # EXIF data exports
│       ├── training_exif_latest.csv
│       └── inference_exif_*.csv
│
├── outputs/                           # Kết quả
│   ├── training_results/             # Kết quả huấn luyện
│   │   ├── 20251210_120000/
│   │   │   ├── classification_report.txt
│   │   │   ├── confusion_matrix.png
│   │   │   ├── feature_importance.png
│   │   │   └── label_distribution.png
│   │   └── ...
│   │
│   └── classified_images/            # Ảnh đã phân loại
│       ├── 20251210_120000/
│       │   ├── classification_results.csv
│       │   ├── Chân Dung/
│       │   │   ├── Sáng/
│       │   │   ├── Chiều/
│       │   │   ├── Tối/
│       │   │   └── ByPerson/
│       │   │       ├── NguyenVanA/
│       │   │       ├── TranThiB/
│       │   │       └── Unknown/
│       │   ├── Tĩnh Vật/
│       │   │   ├── Sáng/
│       │   │   ├── Chiều/
│       │   │   └── Tối/
│       │   ├── Thể Thao/
│       │   ├── Phong Cảnh/
│       │   └── Wildlife/
│       └── ...
│
├── notebooks/                         # Jupyter notebooks (phân tích)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_exif_extractor.py
│   ├── test_face_detector.py
│   └── test_classifier.py
│
└── scripts/                           # Scripts tiện ích
    ├── setup_project.py              # Setup ban đầu
    ├── download_models.py            # Download pre-trained models
    ├── train_model.py                # Script train model
    ├── classify_photos.py            # Script phân loại ảnh
    └── add_known_face.py             # Thêm người vào database
```

##  Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9+
- Windows/Linux/MacOS
- RAM: 8GB+
- GPU (optional, tăng tốc YOLOv8)
- Disk: 35GB+ free space (30GB data + 5GB models/outputs)

### 2. Clone repository

```bash
git clone https://github.com/your-username/photo-classifier.git
cd photo-classifier
```

### 3. Tạo virtual environment và cài dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 4. Setup dự án

```bash
python scripts/setup_project.py
```

### 5. Download training data (30GB)

```bash
# Download từ Google Drive và tự động giải nén
python scripts/download_data.py

# Hoặc download thủ công từ link trong data/README.md
```

### 6. Download pre-trained models (optional)

```bash
# Nếu không muốn train lại, download models có sẵn
python scripts/download_models.py

# Hoặc download từ HuggingFace (xem models/README.md)
```

### 7. Verify installation

```bash
python scripts/verify_dataset.py
```

## 📖 Sử dụng

### 1. Huấn luyện model

```bash
# Train model với dữ liệu mới
python scripts/train_model.py --training-dir "E:\imagee" --output-dir "./outputs/training_results"

# Hoặc dùng Python API
python -c "from src.workflows.training_workflow import train_classifier; train_classifier()"
```

### 2. Phân loại ảnh mới

```bash
# Phân loại ảnh trong folder
python scripts/classify_photos.py --input-dir "./data/inference_images" --output-dir "./outputs/classified_images"

# Phân loại 1 ảnh cụ thể
python scripts/classify_photos.py --image "path/to/photo.jpg"
```

### 3. Thêm người vào database nhận dạng

```bash
python scripts/add_known_face.py --name "NguyenVanA" --photos "path/to/photos/*.jpg"
```

### 4. Sử dụng trong code

```python
from src.workflows.inference_workflow import PhotoClassifier

# Khởi tạo classifier
classifier = PhotoClassifier(
    model_path='./models/random_forest_classifier.pkl',
    face_model_path='./models/yolov8n-face.pt',
    known_faces_db='./data/known_faces'
)

# Phân loại 1 ảnh
result = classifier.classify_single_image('path/to/photo.jpg')
print(result)
# Output: {
#     'category': 'ChanDung',
#     'time_bucket': 'Sáng',
#     'confidence': 0.95,
#     'has_face': True,
#     'recognized_people': ['NguyenVanA'],
#     'destination_path': './outputs/classified_images/.../ChanDung/ByPerson/NguyenVanA/photo.jpg'
# }

# Phân loại hàng loạt
classifier.classify_batch('./data/inference_images')
```

##  Cấu hình

Chỉnh sửa `config.yaml`:

```yaml
# Đường dẫn
paths:
  training_images: "E:/imagee"
  inference_images: "./data/inference_images"
  known_faces: "./data/known_faces"
  models: "./models"
  outputs: "./outputs"

# Model settings
model:
  type: "RandomForest"
  n_estimators: 500
  max_depth: 15
  test_size: 0.1
  random_state: 42

# Face detection
face_detection:
  enabled: true
  model: "yolov8n-face.pt"
  confidence_threshold: 0.5
  min_face_size_ratio: 0.02  # Khuôn mặt phải chiếm ít nhất 2% ảnh

# Face recognition
face_recognition:
  enabled: true
  model: "Facenet512"  # VGG-Face, ArcFace, Facenet512
  distance_threshold: 0.6

# Categories
categories:
  - name: "ChanDung"
    display_name: "Chân Dung"
    keywords: ["chandung", "portrait"]
  - name: "TinhVat"
    display_name: "Tĩnh Vật"
    keywords: ["tinhvat", "stilllife", "macro"]
  - name: "TheThao"
    display_name: "Thể Thao"
    keywords: ["thethao", "sport"]
  - name: "PhongCanh"
    display_name: "Phong Cảnh"
    keywords: ["phongcanh", "landscape"]
  - name: "DongVat"
    display_name: "Wildlife"
    keywords: ["dongvat", "wildlife"]

# Time buckets
time_buckets:
  morning:
    name: "Sáng"
    start_hour: 5
    end_hour: 12
  afternoon:
    name: "Chiều"
    start_hour: 12
    end_hour: 18
  evening:
    name: "Tối"
    start_hour: 18
    end_hour: 5
```

##  Features sử dụng

### EXIF-based Features (10 features)
1. `FocalLength_35mm` - Tiêu cự quy đổi 35mm
2. `FocusDist_Clean` - Khoảng cách lấy nét
3. `DoF_Clean` - Độ sâu trường ảnh
4. `HyperfocalRatio` - Tỷ lệ hyperfocal
5. `FNumber` - Khẩu độ
6. `LogShutterSpeed` - Log tốc độ màn trập
7. `ISO` - Độ nhạy sáng
8. `LightValue` - Giá trị ánh sáng
9. `Hour` - Giờ chụp
10. `Flash_Binary` - Flash có bật không

### Vision-based Features (3 features)
11. `HasFace` - Có khuôn mặt không
12. `NumFaces` - Số lượng khuôn mặt
13. `FaceAreaRatio` - Tỷ lệ diện tích khuôn mặt lớn nhất

##  Độ chính xác

### Trước khi thêm Face Detection (EXIF only)
- Overall Accuracy: **79%**
- ChanDung F1-score: **0.73**
- TinhVat F1-score: **0.65** 

### Sau khi thêm Face Detection (EXIF + Vision)
- Overall Accuracy: **~88-92%**
- ChanDung F1-score: **~0.90-0.95** 
- TinhVat F1-score: **~0.85-0.90** 

##  Testing

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/test_face_detector.py -v

# Test với coverage
pytest --cov=src tests/
```
