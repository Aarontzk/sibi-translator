# sibi-translator

Penerjemah ejaan jari SIBI → teks secara real-time dari webcam. Scope: 24 huruf
statis **A–Y tanpa J** (J dan Z butuh gerak, tidak dicakup), satu tangan.

Pipeline: kamera (OpenCV) → MediaPipe Hands (21 landmark) → normalisasi fitur →
klasifikasi (Random Forest / MLP) → stabilisasi → teks di layar.

## Struktur

```
sibi-translator/
├── collect_dataset.py     # rekam per huruf, dipandu gambar SIBI -> data/raw/landmarks.csv
├── convert_images.py      # foto dataset SIBI Mendeley -> data/raw/sibi_mendeley.csv
├── train_model.py         # latih RF + MLP, simpan terbaik -> models/sibi_model.joblib
├── app_realtime.py        # aplikasi kamera real-time
├── evaluate.py            # classification report + confusion matrix -> reports/
├── benchmark_fps.py       # FPS & waktu per tahap
├── live_test.py / .html   # UI uji langsung di browser (lokal)
├── export_web.py          # ekspor MLP ke web/public/model.json
├── sibi/
│   ├── features.py        # normalize_landmarks (satu-satunya tempat normalisasi)
│   ├── guide.py           # potongan gambar panduan SIBI per huruf
│   ├── stabilizer.py      # satu huruf per isyarat (N-frame + cooldown)
│   └── ui.py              # tampilan (landmark, bar, panel teks)
├── data/raw/              # dataset CSV (koordinat mentah)
├── data/images/           # foto sumber Mendeley (tidak di-commit)
├── models/                # model terlatih (.joblib)
├── reports/               # confusion matrix, hasil uji langsung
├── web/                   # versi web (Vite + React + Elastic UI), di-deploy ke Vercel
└── requirements.txt       # versi terkunci
```

## Pembagian tugas (PIC per deliverable)

| Deliverable | PIC | File terkait |
|---|---|---|
| Dataset | Farel | `collect_dataset.py`, `convert_images.py`, `sibi/guide.py`, `data/raw/` |
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

Urutan: **data → latih → jalankan**. Aplikasi butuh `models/sibi_model.joblib`,
yang dilatih dari `data/raw/sibi_mendeley.csv` (sudah ada di repo) ditambah
rekaman tim di `data/raw/landmarks.csv`.

```bash
# 1. Rekam dataset (tiap anggota, sebutkan nama + tangan)
python collect_dataset.py --person azka --hand right
#    ikuti gambar SIBI di panel kanan, Space = mulai/jeda rekam. Lihat "Cara merekam".

# 2. Latih model (bandingkan RF vs MLP, simpan terbaik ke models/sibi_model.joblib)
python train_model.py

# 2b. Perbarui model versi web (web/public/model.json), lalu deploy ulang folder web/
python export_web.py

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

Ada dua sumber, keduanya SIBI satu tangan (tangan kanan):

| File | Isi | Asal |
|---|---|---|
| `data/raw/sibi_mendeley.csv` | ±1.430 baris, ±60 per huruf | foto dataset SIBI Mendeley, lewat `convert_images.py` |
| `data/raw/landmarks.csv` | rekaman tim | `collect_dataset.py` |

`train_model.py` memakai keduanya. Hold-out-nya adalah folder `test/` milik
penyusun dataset Mendeley, ditambah 20% data tim. Tiap pasangan (orang, huruf)
dibatasi 150 baris supaya satu orang atau satu huruf tidak mendominasi.

### Dataset SIBI Mendeley

Andriyanto, T. *Labeled Official Indonesian Sign Language (SIBI) Alphabet*.
Mendeley Data, v1, 2024. doi:10.17632/yhsgmynccf.1 (CC BY 4.0, **wajib dicantumkan
di References**). Yang dipakai hanya folder `sibi_alphabet_mono_bg_no_augmentation`.
Folder lain berisi foto yang sama dengan latar diganti atau di-blur, jadi
landmark-nya hampir identik dan tidak menambah informasi.

Untuk membuat ulang CSV-nya, unduh folder itu ke `data/images/sibi_mendeley/`
(isinya `train/`, `val/`, `test/`), lalu:

```bash
python convert_images.py data/images/sibi_mendeley
```

Foto tangan kiri dicerminkan jadi tangan kanan. Foto yang tangannya tidak
terdeteksi MediaPipe dilewati (7 dari 1.440).

Dataset video Trunojoyo yang dulu dipakai sudah dibuang karena bentuk tangannya
BISINDO, bukan SIBI (misalnya B hanya telunjuk, Y hanya telunjuk).

### Cara merekam

Acuannya satu: gambar panduan `web/public/abjad-sibi.png`, sama dengan yang
tampil di web. `collect_dataset.py` menampilkan potongan gambar huruf yang sedang
direkam di samping kamera.

1. `python collect_dataset.py --person <nama> --hand right` (pakai nama yang sama tiap sesi).
2. Tiru bentuk tangan di gambar, lalu tekan **Space**. Selama merekam, tahan bentuknya
   sambil pelan-pelan menggeser tangan: maju-mundur, sedikit miring, kiri-kanan layar.
3. Di 100 sampel (`--target`), rekaman berhenti sendiri dan pindah ke huruf berikutnya
   yang belum selesai. **N** lanjut, **B** kembali, **Q** keluar.
4. Bisa dicicil: menjalankan ulang dengan `--person` yang sama melanjutkan huruf yang belum selesai.
5. Kalau salah rekam, hapus baris huruf + nama itu dari `data/raw/landmarks.csv`, lalu rekam ulang.

Rekaman tim menambah variasi tangan di luar dataset Mendeley. Target: semua
anggota merekam ke-24 huruf dengan tangan kanan. Rekam di
beberapa kondisi (terang/redup, latar berbeda) kalau sempat. Dengan 3 perekam atau
lebih, `train_model.py` menguji pada orang yang tidak ikut dilatih, jadi akurasinya
jujur untuk tangan baru.

## Catatan

- `max_num_hands=1`: jika ada dua tangan, hanya satu diproses.
- Model disimpan sebagai `joblib` dict `{"model", "labels"}`; prediksi memakai
  `predict_proba` untuk confidence.
- Self-check cepat tanpa webcam: `python -m sibi.features` dan `python -m sibi.stabilizer`.
