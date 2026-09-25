# SARIMA Forecast Dashboard — Water Park Sumenep

Dashboard Streamlit untuk menampilkan data historis, hasil training/testing, windowing 6→1 dan 12→1, evaluasi model, konfigurasi model, serta forecast masa depan.

## Model
SARIMA(1,0,1)(0,0,0,12)

## Data wajib
Salin 7 file CSV hasil ekspor final ke folder `streamlit_data/`:

- `data_historis.csv`
- `hasil_training.csv`
- `hasil_testing.csv`
- `window_6.csv`
- `window_12.csv`
- `metrics.csv`
- `konfigurasi_model.csv`

CSV final yang tervalidasi menggunakan 108 observasi dan tidak membuat data tahun 2020 secara sintetis.

## Menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload seluruh isi folder ini ke repository GitHub bersama folder `streamlit_data/`, kemudian deploy `app.py` melalui Streamlit Community Cloud.
