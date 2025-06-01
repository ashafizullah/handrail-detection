# Handrail Detection System

Sistem deteksi otomatis untuk memantau apakah karyawan memegang handrail saat naik atau turun tangga kantor.

## Fitur

- **Deteksi Pose Manusia**: Menggunakan MediaPipe untuk mendeteksi posisi tangan
- **Deteksi Handrail**: Deteksi pegangan tangan menggunakan edge detection dan line detection
- **Analisis Proximity**: Menghitung jarak antara tangan dan handrail
- **Visualisasi Real-time**: Menampilkan hasil deteksi secara langsung
- **Laporan Keamanan**: Generate laporan compliance keamanan

## Instalasi

1. Clone repository ini
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Penggunaan

### Basic Usage
```bash
python src/main.py path/to/video.mp4
```

### Dengan Output Video
```bash
python src/main.py input.mp4 --output output_analyzed.mp4
```

### Tanpa Display (untuk server)
```bash
python src/main.py input.mp4 --no-display --output result.mp4
```

### Custom Threshold
```bash
python src/main.py input.mp4 --threshold 40 --output result.mp4
```

## Parameter

- `--output, -o`: Path untuk menyimpan video hasil analisis
- `--threshold, -t`: Threshold jarak untuk deteksi sentuhan (default: 30 pixel)
- `--no-display`: Nonaktifkan tampilan video real-time

## Struktur Project

```
handrail-detection/
├── src/
│   ├── detection/
│   │   ├── pose_detector.py      # Deteksi pose manusia
│   │   ├── handrail_detector.py  # Deteksi handrail
│   │   └── proximity_analyzer.py # Analisis kedekatan
│   ├── utils/
│   │   ├── video_processor.py    # Pemrosesan video
│   │   └── visualization.py      # Visualisasi hasil
│   └── main.py                   # Script utama
├── models/                       # Model tambahan (jika ada)
├── data/
│   ├── videos/                   # Video input
│   └── output/                   # Hasil analisis
└── requirements.txt
```

## Output

- **Video beranotasi**: Video dengan overlay deteksi pose, handrail, dan status keamanan
- **Laporan konsol**: Statistik compliance keamanan
- **Summary plot**: Grafik penggunaan handrail sepanjang video (disimpan di `data/output/`)

## Cara Kerja

1. **Pose Detection**: Sistem mendeteksi keypoints tubuh manusia, fokus pada posisi pergelangan tangan
2. **Handrail Detection**: Menggunakan edge detection dan HoughLines untuk mendeteksi garis horizontal yang merupakan handrail
3. **Proximity Analysis**: Menghitung jarak perpendicular dari tangan ke handrail terdekat
4. **Classification**: Jika jarak < threshold, dianggap sedang memegang handrail

## Contoh Video Input

Letakkan video Anda di folder `data/videos/` dan jalankan:
```bash
python src/main.py data/videos/stairs_video.mp4 --output data/output/analyzed_video.mp4
```

## Troubleshooting

- Pastikan video input memiliki resolusi yang cukup untuk deteksi pose
- Untuk hasil optimal, handrail harus terlihat jelas dan kontras dengan background
- Adjust threshold sesuai dengan resolusi video dan jarak kamera ke subjek