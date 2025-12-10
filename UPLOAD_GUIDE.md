# Photo Classifier - GitHub Upload Guide

## Vấn đề: Dataset 30GB không thể upload

GitHub giới hạn:
- 100MB/file
- 1GB/repo total
- Dataset của bạn: 30GB ❌

## Giải pháp

### 1. Upload Code lên GitHub (chỉ ~5MB)

```bash
cd photo_classifier_project

# Init git (nếu chưa có)
git init

# Add remote
git remote add origin https://github.com/your-username/photo-classifier.git

# Add files (models và data đã bị ignore)
git add .
git commit -m "Initial commit: Photo classifier with EXIF + YOLOv8"
git push -u origin main
```

### 2. Models (~50MB) - 3 options:

#### A. Git LFS (Large File Storage) - Khuyến nghị
```bash
# Cài Git LFS
git lfs install

# Track models
git lfs track "models/*.pkl"
git lfs track "models/*.pt"

# Commit
git add .gitattributes
git add models/
git commit -m "Add trained models via LFS"
git push

# Note: GitHub LFS free = 1GB storage, 1GB bandwidth/month
```

#### B. Google Drive / Dropbox
```bash
# Upload models lên Google Drive
# Share link public
# User download thủ công và đặt vào models/
```

#### C. HuggingFace Hub (Best for ML)
```python
# Upload
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="models/random_forest_classifier.pkl",
    path_in_repo="random_forest_classifier.pkl",
    repo_id="your-username/photo-classifier",
    token="your_token"
)

# Download (tích hợp vào code)
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="your-username/photo-classifier",
    filename="random_forest_classifier.pkl"
)
```

### 3. Data (30GB) - Upload lên Google Drive

**Các bước:**

1. **Nén dataset:**
```bash
# Windows
Compress-Archive -Path "data\training_images" -DestinationPath "training_data.zip"

# Linux/Mac
zip -r training_data.zip data/training_images/
```

2. **Upload lên Google Drive:**
   - Truy cập drive.google.com
   - Upload file ZIP (~8-10GB sau nén)
   - Right click file → Share → Anyone with the link → Copy link
   - Lấy FILE_ID từ link: `https://drive.google.com/file/d/FILE_ID_HERE/view`

3. **Cập nhật FILE_ID:**
```python
# Sửa file: scripts/download_data.py
GOOGLE_DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"  # Paste FILE_ID vào đây
```

4. **Commit code (không có data):**
```bash
git add .
git commit -m "Add Google Drive download script"
git push
```

**Users sẽ download data:**
```bash
python scripts/download_data.py
```

## Files được upload lên GitHub:

```
✅ src/                    # Source code
✅ scripts/                # Scripts
✅ config.yaml             # Config
✅ requirements.txt        # Dependencies
✅ README.md               # Documentation
✅ .gitignore              # Ignore rules
✅ models/.gitkeep         # Placeholder
✅ models/README.md        # Download instructions
✅ data/.gitkeep           # Placeholder
✅ data/README.md          # Dataset instructions

❌ models/*.pkl           # Ignored (dùng LFS hoặc external)
❌ models/*.pt            # Ignored
❌ data/training_images/  # Ignored (30GB)
❌ data/metadata/*.csv    # Ignored
❌ outputs/               # Ignored
```

## Recommended: HuggingFace Hub

**Tại sao:**
- Free unlimited storage cho models
- Versioning tự động
- Easy integration với Python
- Community trust (như GitHub cho ML)

**Setup:**
```bash
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
python scripts/upload_to_huggingface.py
```

## Kết quả cuối:

- **GitHub**: Code + Documentation (~5MB)
- **HuggingFace Hub**: Trained models (~50MB)
- **User's machine**: Training data (30GB) - tự chuẩn bị

Users clone GitHub repo → Download models từ HuggingFace → Train lại hoặc dùng pre-trained models.
