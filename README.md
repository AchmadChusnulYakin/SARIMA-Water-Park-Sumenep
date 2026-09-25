# 📊 SARIMA Forecast Dashboard — Water Park Sumenep

Dashboard interaktif berbasis **Streamlit** untuk menampilkan hasil pemodelan dan peramalan jumlah kunjungan wisatawan pada **Water Park Sumenep** menggunakan model **SARIMA**.

Project ini merupakan implementasi **baseline forecasting** dalam kegiatan **MBKM Riset** dan digunakan untuk menyajikan hasil pengolahan data, evaluasi model, skenario windowing, serta forecast masa depan.

---

## 📌 Ringkasan Penelitian

Dataset yang digunakan terdiri dari **108 observasi** jumlah kunjungan wisatawan dengan tanggal yang tersedia dari **Januari 2015 sampai Desember 2024**.

Pada tahap preprocessing terdapat **4 nilai nol** yang, sesuai arahan pembimbing, diperlakukan sebagai nilai yang perlu ditangani. Nilai tersebut diubah menjadi `NaN` dan kemudian diestimasi menggunakan **interpolasi linear**.

**Tahun 2020 tidak tersedia pada dataset mentah dan tidak dibuat atau diisi secara sintetis**, sehingga jumlah observasi penelitian tetap 108 data.

### Konfigurasi utama

| Komponen | Nilai |
|---|---:|
| Jumlah observasi | 108 |
| Nilai nol yang ditangani | 4 |
| Training | 75 observasi (70%) |
| Testing | 33 observasi (30%) |
| Window 6 → 1 | 102 observasi |
| Window 12 → 1 | 96 observasi |
| Model | SARIMA(1,0,1)(0,0,0,12) |
| Periode musiman | 12 bulan |
| Evaluasi | MAE, RMSE, MAPE, R² |

Karena `P=D=Q=0`, model tersebut secara matematis setara dengan **ARIMA(1,0,1)** tanpa komponen musiman aktif, meskipun dituliskan dalam kerangka SARIMA dengan `s=12`.

---

## 🔬 Alur Pengolahan

```text
Dataset
   ↓
Pemeriksaan Struktur Data
   ↓
Penanganan Nilai 0
   ↓
Interpolasi Linear
   ↓
Uji Stasioneritas ADF
   ↓
Identifikasi ACF & PACF
   ↓
Penentuan Parameter SARIMA
   ↓
Pembagian Training / Testing
   ↓
Pemodelan SARIMA
   ↓
Evaluasi MAE, RMSE, MAPE, R²
   ↓
Rolling Forecast Window 6 → 1 & 12 → 1
   ↓
Ekspor Hasil ke CSV
   ↓
Dashboard Streamlit
```

---

## 📈 Hasil Evaluasi

| Skenario | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Training | 4660.1480 | 7804.3088 | 139.7986% | -0.0003 |
| Testing | 1166.3402 | 1368.5091 | 176.3118% | -0.0100 |
| Window 6 → 1 | 4822.6281 | 10429.5463 | 171.7417% | -2.0276 |
| Window 12 → 1 | 2998.8691 | 6186.0886 | 131.2170% | -0.4283 |

Nilai tersebut merupakan hasil evaluasi yang digunakan sebagai **baseline SARIMA** dalam penelitian.

---

## 🖥️ Fitur Dashboard

### 🏠 Beranda
Ringkasan dataset, jumlah training/testing, periode data, dan visualisasi historis.

### 📈 Data Historis
Tabel dan visualisasi data kunjungan setelah preprocessing.

### 🔮 Hasil SARIMA
Perbandingan nilai **Aktual** dan **Prediksi** untuk training dan testing.

### 🪟 Windowing
Menampilkan rolling forecast untuk:
- Window 6 bulan → 1 prediksi
- Window 12 bulan → 1 prediksi

### 📅 Forecast Masa Depan
Menghasilkan prediksi beberapa bulan setelah observasi historis terakhir menggunakan model SARIMA, termasuk **interval prediksi 95%**.

### 📊 Evaluasi Model
Menampilkan MAE, RMSE, MAPE, dan R² dalam bentuk tabel dan visualisasi.

### ⚙️ Konfigurasi Model
Menampilkan struktur:
```text
SARIMA(1,0,1)(0,0,0,12)
```

---

## 📁 Struktur Repository

```text
SARIMA-Water-Park-Sumenep/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── Streamlit Data/
    ├── data_historis.csv
    ├── hasil_training.csv
    ├── hasil_testing.csv
    ├── window_6.csv
    ├── window_12.csv
    ├── metrics.csv
    └── konfigurasi_model.csv
```

> Nama folder data harus sama dengan path yang digunakan pada `app.py`, yaitu `Streamlit Data`.

---

## 📦 File Data

- `data_historis.csv` — data historis setelah preprocessing dan interpolasi linear.
- `hasil_training.csv` — aktual dan prediksi pada 75 observasi training.
- `hasil_testing.csv` — aktual dan prediksi pada 33 observasi testing.
- `window_6.csv` — rolling forecast dengan histori 6 observasi untuk 1 langkah prediksi.
- `window_12.csv` — rolling forecast dengan histori 12 observasi untuk 1 langkah prediksi.
- `metrics.csv` — MAE, RMSE, MAPE, dan R².
- `konfigurasi_model.csv` — konfigurasi model SARIMA.

---

## 🛠️ Teknologi

- Python
- Pandas
- NumPy
- Statsmodels
- Plotly
- Streamlit

Model:
```text
SARIMA(1,0,1)(0,0,0,12)
```

---

## ▶️ Menjalankan Secara Lokal

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan dashboard:

```bash
streamlit run app.py
```

Dashboard biasanya tersedia di:

```text
http://localhost:8501
```

---

## ☁️ Deployment

Project disiapkan untuk deployment menggunakan **Streamlit Community Cloud** melalui repository GitHub.

File entrypoint:
```text
app.py
```

Dependency:
```text
requirements.txt
```

Folder data:
```text
Streamlit Data/
```

Pastikan seluruh 7 CSV final berada di repository sebelum deployment.

---

## 🧪 Catatan Metodologis

Dashboard ini merupakan implementasi **baseline SARIMA**.

Hasil **Window 6 → 1** dan **Window 12 → 1** merupakan skenario evaluasi rolling forecast dan berbeda dari fitur **Forecast Masa Depan**, yang digunakan untuk menghasilkan prediksi setelah periode historis terakhir.

Karena tahun 2020 tidak tersedia dalam dataset mentah, terdapat gap kalender pada visualisasi. Gap tersebut **tidak berarti data tahun 2020 dibuat atau diinterpolasi**.

---

## 👤 Project

**Dashboard SARIMA — Water Park Sumenep**

Bagian dari kegiatan:

**MBKM Skema Penelitian/Riset — Pemodelan SARIMA untuk Peramalan Kunjungan Wisatawan Berbasis Data Runtun Waktu**

---

## 📄 Lisensi

Project ini dibuat untuk kebutuhan akademik dan dokumentasi penelitian.
