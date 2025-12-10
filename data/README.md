# Training Data (30GB)

## Download từ Google Drive

Dataset đầy đủ (30GB+) được lưu trữ trên Google Drive:

**Download link:** [Sẽ cập nhật sau khi upload]

### Cấu trúc sau khi giải nén:

```
data/
├── training_images/          # 30GB training data
│   ├── ChanDung/            # ~800 portrait photos
│   ├── TinhVat/             # ~500 still life photos
│   ├── TheThao/             # ~300 sports photos
│   ├── PhongCanh/           # ~900 landscape photos
│   └── DongVat/             # ~200 wildlife photos
│
├── metadata/                 # EXIF exports (CSV)
│   └── exif_data_complete.csv
│
└── sample_images/           # Sample images để test (50 ảnh)
```

## Hướng dẫn cài đặt

### 1. Download dataset
```bash
# Option A: Download thủ công
1. Truy cập Google Drive link ở trên
2. Download file ZIP (~8-10GB sau nén)
3. Giải nén vào thư mục data/

# Option B: Dùng gdown (tự động)
pip install gdown
python scripts/download_data.py
```

### 2. Kiểm tra cấu trúc
```bash
python scripts/verify_dataset.py
```

## Dataset Info

- **Total images:** ~2,700 ảnh
- **Total size:** 30GB (RAW files)
- **Formats:** CR3, ARW, NEF, JPG
- **EXIF:** Đầy đủ metadata
- **Categories:** 5 loại (ChanDung, TinhVat, TheThao, PhongCanh, DongVat)

## Tự chuẩn bị dataset (Optional)

Nếu muốn dùng data riêng:
1. Tổ chức theo cấu trúc trên
2. Đảm bảo ảnh có EXIF đầy đủ
3. Khuyến nghị: 100-200 ảnh/category minimum
