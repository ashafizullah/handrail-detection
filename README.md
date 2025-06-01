# Handrail Detection System

Sistem deteksi otomatis untuk memantau apakah karyawan memegang handrail saat naik atau turun tangga kantor menggunakan MediaPipe dan computer vision.

## 🎯 Fitur Utama

- **🤖 Multi-Person Detection**: Deteksi beberapa orang dalam satu frame
- **🎨 Manual Annotation**: Anotasi manual handrail untuk akurasi maksimal
- **📊 Real-time Analysis**: Visualisasi deteksi secara langsung
- **📈 Compliance Report**: Laporan keamanan dan statistik lengkap
- **🔄 Fixed Position**: Sistem anotasi sekali pakai untuk seluruh video

## 📋 Prerequisites

- Python 3.11 (untuk MediaPipe compatibility)
- OpenCV, MediaPipe, NumPy
- Virtual environment (recommended)

## 🚀 Installation

### Step 1: Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd handrail-detection

# Install Python 3.11 (jika belum ada)
brew install python@3.11

# Buat virtual environment dengan Python 3.11
python3.11 -m venv handrail-env-mediapipe
source handrail-env-mediapipe/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Video
```bash
# Letakkan video di folder data/videos/
cp your_video.mp4 data/videos/
```

## 📖 Cara Penggunaan

### 🎯 Method 1: Quick Annotation (RECOMMENDED)

**Untuk akurasi maksimal dengan anotasi manual sekali saja:**

```bash
# One-command complete workflow
python3 quick_annotate.py data/videos/your_video.mp4
```

**Workflow ini akan:**
1. 🎨 Membuka annotation tool untuk menggambar handrail di frame 0
2. 🔄 Otomatis apply posisi handrail ke semua frame
3. 🚀 Menjalankan analisis dengan handrail tetap
4. 📹 Menghasilkan video hasil dan laporan

### 🎨 Method 2: Manual Step-by-Step

#### Step 1: Activate Environment
```bash
source handrail-env-mediapipe/bin/activate
```

#### Step 2: Create Manual Annotations (Frame 0 Only)
```bash
python annotate_video.py data/videos/your_video.mp4
```

**Controls saat annotation:**
- **Left click + drag**: Gambar garis handrail
- **Right click**: Hapus handrail terakhir
- **SPACE**: Frame berikutnya (optional)
- **'s'**: Simpan annotations
- **'q'**: Keluar dan simpan

#### Step 3: Apply to All Frames
```bash
python3 apply_fixed_annotations.py data/videos/your_video_annotations.json
```

#### Step 4: Run Analysis
```bash
python src/main.py data/videos/your_video.mp4 \
  --annotations data/videos/your_video_annotations_fixed_all_frames.json \
  --output data/output/your_video_analyzed.mp4
```

### ⚡ Method 3: Automatic Detection (No Annotations)

**Untuk testing cepat tanpa anotasi manual:**

```bash
# Basic usage
source handrail-env-mediapipe/bin/activate
python src/main.py data/videos/your_video.mp4 --output data/output/result.mp4

# Dengan custom threshold
python src/main.py data/videos/your_video.mp4 --threshold 40 --output result.mp4

# Tanpa display (untuk server)
python src/main.py data/videos/your_video.mp4 --no-display --output result.mp4
```

## ⚙️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output, -o` | Path output video | None |
| `--annotations, -a` | Path file annotations JSON | None |
| `--threshold, -t` | Threshold jarak sentuhan (pixels) | 30 |
| `--no-display` | Nonaktifkan tampilan real-time | False |

## 📊 Output yang Dihasilkan

### 📹 Video Results
- **Annotated video**: Video dengan overlay detection
- **Bounding boxes**: Kotak di sekitar setiap orang
- **Handrail lines**: Garis handrail yang terdeteksi
- **Safety status**: Status real-time per person

### 📈 Analysis Report
```
============================================================
HANDRAIL DETECTION ANALYSIS REPORT
============================================================
Detection method: Manual Annotation
Total frames analyzed: 192
Frame-based safety compliance: 60.9%
Average handrails detected: 5.0

PEOPLE COUNTING ANALYSIS:
------------------------------
Maximum people detected in any frame: 6
Average people per frame: 2.5
People instances using handrail: 121
People instances NOT using handrail: 351
People-based safety compliance: 25.6%
```

### 📂 File Output
- `your_video_analyzed.mp4` - Video dengan analisis
- `handrail_analysis_summary.png` - Grafik summary
- `your_video_annotations.json` - Anotasi original
- `your_video_annotations_fixed_all_frames.json` - Anotasi untuk semua frame

## 🏗️ Struktur Project

```
handrail-detection/
├── src/
│   ├── detection/
│   │   ├── pose_detector_mediapipe.py    # Multi-person pose detection
│   │   ├── handrail_detector.py          # Automatic handrail detection
│   │   ├── annotation_based_detector.py  # Manual annotation detector
│   │   ├── proximity_analyzer.py         # Distance analysis
│   │   └── people_tracker.py             # Person tracking
│   ├── utils/
│   │   ├── manual_annotation.py          # Annotation tool
│   │   ├── video_processor.py            # Video processing
│   │   └── visualization.py              # Results visualization
│   └── main.py                           # Main analysis script
├── data/
│   ├── videos/                           # Input videos
│   └── output/                           # Analysis results
├── models/                               # Model files
├── annotate_video.py                     # Annotation tool launcher
├── quick_annotate.py                     # One-command workflow
├── apply_fixed_annotations.py            # Apply annotations to all frames
└── requirements.txt                      # Dependencies
```

## 🔧 Advanced Usage

### Custom Annotation Workflow
```bash
# Untuk video dengan multiple scenes
python annotate_video.py data/videos/complex_video.mp4
# Annotate frame 0, 50, 100, 150 (key frames)

# Apply dengan interpolation
python3 apply_fixed_annotations.py data/videos/complex_video_annotations.json

# Run analysis
python src/main.py data/videos/complex_video.mp4 \
  --annotations data/videos/complex_video_annotations_fixed_all_frames.json \
  --threshold 35 \
  --output data/output/complex_analyzed.mp4
```

### Batch Processing
```bash
# Process multiple videos
for video in data/videos/*.mp4; do
    echo "Processing $video"
    python3 quick_annotate.py "$video"
done
```

## 📊 Detection Methods Comparison

| Method | Accuracy | Setup Time | Consistency | Use Case |
|--------|----------|------------|-------------|----------|
| **Manual Annotation** | 🟢 95%+ | 🟡 5-10 min | 🟢 Perfect | Production |
| **Automatic Detection** | 🟡 70-80% | 🟢 Instant | 🟡 Variable | Quick test |

## 🚨 Troubleshooting

### Common Issues

**1. MediaPipe Import Error**
```bash
# Solution: Use Python 3.11
python3.11 -m venv handrail-env-mediapipe
source handrail-env-mediapipe/bin/activate
pip install mediapipe
```

**2. Annotation Tool Crashes**
```bash
# Solution: Activate environment first
source handrail-env-mediapipe/bin/activate
python annotate_video.py your_video.mp4
```

**3. Low Detection Accuracy**
```bash
# Solution: Use manual annotations
python3 quick_annotate.py your_video.mp4
```

**4. Video Not Found**
```bash
# Solution: Check path
ls data/videos/
python src/main.py data/videos/existing_video.mp4
```

### Performance Tips

- **Video Resolution**: 720p-1080p optimal
- **Handrail Visibility**: Pastikan handrail kontras dengan background
- **Lighting**: Hindari backlight atau shadow yang berlebihan
- **Camera Angle**: Front/side view lebih baik dari top-down

## 📞 Support

Jika mengalami masalah:
1. Check video di `data/videos/` directory
2. Pastikan virtual environment aktif
3. Verify Python 3.11 installation
4. Lihat troubleshooting section

---

**🎯 Recommended Workflow**: Gunakan `python3 quick_annotate.py` untuk hasil terbaik dengan minimal effort!