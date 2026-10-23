# sibi-translator

Penerjemah ejaan jari SIBI → teks secara real-time dari webcam. Scope: 24 huruf
statis **A–Y tanpa J** (J dan Z butuh gerak, tidak dicakup), satu tangan.

Pipeline: kamera (OpenCV) → MediaPipe Hands (21 landmark) → normalisasi fitur →
klasifikasi (Random Forest / MLP) → stabilisasi → teks di layar.

## Struktur

```
sibi-translator/
├── collect_dataset.py     # rekam landmark per huruf dari webcam -> data/raw/landmarks.csv
├── convert_videos.py      # dataset video luar (Trunojoyo) -> data/raw/external_*.csv
├── train_model.py         # latih RF + MLP, simpan terbaik -> models/sibi_model.joblib
├── app_realtime.py        # aplikasi kamera real-time
├── evaluate.py            # classification report + confusion matrix -> reports/
├── benchmark_fps.py       # FPS & waktu per tahap
├── live_test.py / .html   # UI uji langsung di browser (lokal)
├── export_web.py          # ekspor MLP ke web/public/model.json
├── sibi/
│   ├── features.py        # normalize_landmarks (satu-satunya tempat normalisasi)
│   ├── stabilizer.py      # satu huruf per isyarat (N-frame + cooldown)
│   └── ui.py              # tampilan (landmark, bar, panel teks)
├── data/raw/              # dataset CSV (koordinat mentah)
├── models/                # model terlatih (.joblib)
├── reports/               # confusion matrix, hasil uji langsung
├── web/                   # versi web (Vite + React + Elastic UI), di-deploy ke Vercel
└── requirements.txt       # versi terkunci
```

## Pembagian tugas (PIC per deliverable)

| Deliverable | PIC | File terkait |
|---|---|---|
| Dataset | Farel | `collect_dataset.py`, `convert_videos.py`, `data/raw/` |
| Model klasifikasi | Azka | `train_model.py`, `sibi/features.py`, `models/` |
| Aplikasi real-time | Gregorius | `app_realtime.py`, `sibi/stabilizer.py` |
| Tampilan | Sahril | `sibi/ui.py` |
| Pengujian & evaluasi | Rafi | `evaluate.py`, `benchmark_fps.py`, `live_test.py`, `reports/` |
| Source code & GitHub | Azka | repo ini (struktur, README, merge PR) |
| Laporan & upload | Sahril | `reports/` (bahan hasil dari Rafi dan Azka) |
| Slide | Rafi | - |
| Video demo | Gregorius | rekaman `app_realtime.py` |
| Appendix A & B | Farel | - |

## Setup

```bash
python -m venv venv
# Windows:  venv\Scripts\activate    |    macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

Versi dikunci (`mediapipe`, `opencv-python`, `scikit-learn`, …) agar `sibi_model.joblib`
hasil training bisa dibuka di semua laptop.

## Cara menjalankan

Urutan: **rekam data → latih → jalankan**. Aplikasi butuh `models/sibi_model.joblib`,
yang butuh `data/raw/landmarks.csv`.

```bash
# 1. Rekam dataset (tiap anggota, sebutkan nama + tangan)
python collect_dataset.py --person azka --hand right
#    arahkan tangan ke kamera, tekan huruf (A–Y, tanpa J) untuk menyimpan sampel.

# 2. Latih model (bandingkan RF vs MLP, simpan terbaik ke models/sibi_model.joblib)
python train_model.py

# 3. Jalankan penerjemah real-time
python app_realtime.py

# Evaluasi & benchmark (butuh model + data)
python evaluate.py
python benchmark_fps.py --seconds 60
```

### Uji langsung di browser

```bash
python live_test.py            # lalu buka http://127.0.0.1:8000
```

Pipeline-nya sama persis dengan `app_realtime.py`, ditampilkan di browser.

- **Uji**: isi nama penguji dan skenario (normal / redup / latar ramai / jauh). Tiap huruf diuji N kali (default 10) dalam urutan acak. Satu trial selesai saat stabilizer mengeluarkan huruf atau saat waktunya habis. Hasil otomatis tersimpan ke `reports/live_test/<penguji>_<skenario>_<waktu>.csv` (kolom `tester, scenario, target, predicted, correct, ms`) dan ringkasannya ditampilkan: akurasi, akurasi per huruf, dan pasangan huruf yang tertukar.
- **Bebas**: eja kata seperti di `app_realtime.py`. Tombolnya Space, Backspace, dan C.
- Server hanya bisa diakses dari laptop sendiri (127.0.0.1). Opsi `--camera N` untuk memilih kamera lain.

### Tombol (app_realtime.py)

| Tombol | Fungsi |
|--------|--------|
| `Space` | spasi |
| `Backspace` | hapus huruf terakhir |
| `C` | kosongkan teks |
| `Q` | keluar |

## Dataset

`data/raw/landmarks.csv` berisi koordinat **mentah** MediaPipe (belum dinormalisasi;
normalisasi hanya di `sibi.features.normalize_landmarks`). Kolom:

```
label, person, hand, x0, y0, z0, …, x20, y20, z20
```

Direkam sendiri oleh anggota tim dengan izin. Protokol: variasi cahaya
(terang/redup), jarak (dekat/jauh), sedikit rotasi; target sampel per huruf per
orang disepakati tim. Satu tangan (tangan kiri menghasilkan landmark tercermin —
putuskan sebelum merekam).

### Dataset eksternal (opsional, tambahan)

Dataset video luar dikonversi dengan `convert_videos.py` ke format kolom yang
sama, disimpan **terpisah** dari `landmarks.csv`:

```bash
python convert_videos.py path/ke/video --source trunojoyo --start 0.5 --every 2
# -> data/raw/external_train.csv, data/raw/external_test.csv

python train_model.py --external-only   # latih HANYA dari data eksternal (sebelum tim merekam)
python train_model.py --with-external   # data tim + external_train
python evaluate.py --external     # uji di external_test -> reports/confusion_matrix_external.png
```

`--start 0.5` membuang paruh awal tiap video. Di dataset Trunojoyo, bagian awal
video berisi tangan yang masih naik dan bentuknya belum jadi. Dataset Trunojoyo
sudah dicek: **SIBI**, satu tangan, 640×480, 25 fps. Strukturnya `A/A_001.MOV`, …

Data eksternal hanya ditambahkan ke bagian **latih**. Hold-out tim tetap murni
data tim, jadi `python evaluate.py` biasa tetap bisa dibandingkan antar-run.

- Frame di-flip seperti `app_realtime.py`, supaya landmark tidak tercermin.
- Split 80:20 **per video**, bukan per frame, supaya tidak bocor. Kolom `person` berisi `<source>/<nama video>`.
- Huruf J dan Z dilewati. Label diambil dari nama folder (`A/`) atau huruf pertama nama file.
- Wajib dicantumkan di References (lisensi CC BY 4.0):
  - Siradjuddin, I. *Indonesian Sign Language Dataset: Alphabet Video*. Mendeley Data, v1, 2024. doi:10.17632/p7j5jrsbbb.1
  - Darmawan, I D. M. B. A. et al. *Indonesian Sign Language System (SIBI) Dataset*. Mendeley Data, v3, 2024. doi:10.17632/44pbrbsnkh.3 (versi lengkap butuh Data Use Agreement)

## Catatan

- `max_num_hands=1`: jika ada dua tangan, hanya satu diproses.
- Model disimpan sebagai `joblib` dict `{"model", "labels"}`; prediksi memakai
  `predict_proba` untuk confidence.
- Self-check cepat tanpa webcam: `python -m sibi.features` dan `python -m sibi.stabilizer`.
