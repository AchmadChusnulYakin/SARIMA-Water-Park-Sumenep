import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SARIMA Forecast Dashboard - Waterpark Sumenep",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("Streamlit Data")

# =========================================================
# CUSTOM CSS — modern / professional presentation
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f3a 0%, #102d52 100%);
    }

    [data-testid="stSidebar"] * {
        color: #eef5ff !important;
    }

    .brand {
        padding: 8px 0 20px 0;
    }

    .brand-title {
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #b9cae4 !important;
        line-height: 1.45;
    }

    .hero {
        background: linear-gradient(135deg, #0b1f3a 0%, #153e6d 55%, #225c8f 100%);
        border-radius: 22px;
        padding: 30px 34px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 14px 35px rgba(11, 31, 58, .16);
    }

    .hero-kicker {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .12em;
        opacity: .8;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 33px;
        font-weight: 800;
        line-height: 1.16;
        margin: 0;
    }

    .hero-text {
        font-size: 14px;
        opacity: .9;
        line-height: 1.6;
        max-width: 900px;
        margin-top: 12px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #14253d;
        margin: 10px 0 4px 0;
    }

    .section-subtitle {
        color: #62748a;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .kpi-card {
        background: white;
        border: 1px solid #e4eaf2;
        border-radius: 17px;
        padding: 18px 18px 16px 18px;
        min-height: 116px;
        box-shadow: 0 7px 20px rgba(26, 47, 78, .06);
    }

    .kpi-label {
        color: #6a7d93;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .kpi-value {
        color: #13243c;
        font-size: 27px;
        font-weight: 800;
        margin-top: 7px;
    }

    .kpi-help {
        color: #8090a3;
        font-size: 11px;
        margin-top: 4px;
    }

    .card {
        background: white;
        border: 1px solid #e4eaf2;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 7px 20px rgba(26, 47, 78, .05);
        margin-bottom: 16px;
    }

    .mini-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #eaf3ff;
        color: #1d5b97;
        font-size: 11px;
        font-weight: 800;
    }

    .metric-explain {
        font-size: 12px;
        color: #697b90;
        line-height: 1.5;
    }

    .footer {
        margin-top: 35px;
        padding: 18px 0 8px 0;
        color: #8291a4;
        font-size: 11px;
        border-top: 1px solid #e2e8f0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4eaf2;
        padding: 13px 15px;
        border-radius: 15px;
        box-shadow: 0 7px 18px rgba(26, 47, 78, .05);
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 44px;
    }

    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# DATA LOADING
# =========================================================
REQUIRED_FILES = {
    "historis": "data_historis.csv",
    "training": "hasil_training.csv",
    "testing": "hasil_testing.csv",
    "window6": "window_6.csv",
    "window12": "window_12.csv",
    "metrics": "metrics.csv",
    "config": "konfigurasi_model.csv",
}


def require_files():
    missing = [f for f in REQUIRED_FILES.values() if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "File berikut belum ditemukan di folder streamlit_data/: "
            + ", ".join(missing)
        )


@st.cache_data

def load_data():
    require_files()
    dfs = {}
    for key, filename in REQUIRED_FILES.items():
        dfs[key] = pd.read_csv(DATA_DIR / filename)

    date_cols = ["historis", "training", "testing", "window6", "window12"]
    for key in date_cols:
        if "Tanggal" in dfs[key].columns:
            dfs[key]["Tanggal"] = pd.to_datetime(dfs[key]["Tanggal"], errors="coerce")
            dfs[key] = dfs[key].sort_values("Tanggal").reset_index(drop=True)

    if "Jumlah_Kunjungan" in dfs["historis"].columns:
        dfs["historis"]["Jumlah_Kunjungan"] = pd.to_numeric(
            dfs["historis"]["Jumlah_Kunjungan"], errors="coerce"
        )

    # Validasi integritas sumber dashboard agar CSV versi lama
    # (misalnya 120 observasi dengan tahun 2020 sintetis) tidak terbaca diam-diam.
    expected_rows = {
        "historis": 108,
        "training": 75,
        "testing": 33,
        "window6": 102,
        "window12": 96,
        "metrics": 4,
    }
    for key, expected in expected_rows.items():
        if len(dfs[key]) != expected:
            raise ValueError(
                f"Data {REQUIRED_FILES[key]} tidak sesuai hasil notebook: "
                f"diharapkan {expected} baris, terbaca {len(dfs[key])}. "
                "Ekspor ulang CSV dari notebook versi tervalidasi."
            )

    if 2020 in set(dfs["historis"]["Tanggal"].dt.year):
        raise ValueError(
            "data_historis.csv mengandung tahun 2020. "
            "Dashboard membutuhkan CSV final 108 observasi tanpa membuat data 2020 sintetis."
        )

    return dfs


try:
    data = load_data()
except Exception as exc:
    st.error(f"Tidak dapat memuat data: {exc}")
    st.stop()

historis = data["historis"]
training = data["training"]
testing = data["testing"]
window6 = data["window6"]
window12 = data["window12"]
metrics = data["metrics"]
config = data["config"]

# =========================================================
# HELPERS
# =========================================================

def fmt_num(value, digits=0):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_month(dt):
    if pd.isna(dt):
        return "—"
    bulan = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ]
    return f"{bulan[dt.month - 1]} {dt.year}"


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def metric_value(row, candidates):
    for c in candidates:
        if c in row.index:
            value = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(value):
                return float(value)
    return np.nan


def chart_layout(fig, title=None, height=430):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=18, r=18, t=55 if title else 20, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e8edf3", zeroline=False),
    )
    return fig


def add_kpi(label, value, help_text=""):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-help'>{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_model_params():
    # Parameter utama yang digunakan pada dashboard mengikuti konfigurasi
    # SARIMA yang dipakai pada hasil notebook saat ini.
    return (1, 0, 1), (0, 0, 0), 12


def build_forecast(series, periods):
    # Tahun 2020 tidak tersedia pada data mentah dan tidak dibuat secara sintetis.
    (p, d, q), (P, D, Q), s = get_model_params()

    # Dataset memiliki gap kalender karena tahun 2020 tidak tersedia.
    # Gunakan index numerik di dalam SARIMA agar forecast tidak gagal.
    model_series = pd.Series(
        series.to_numpy(dtype=float),
        index=pd.RangeIndex(len(series)),
        name="Jumlah_Kunjungan"
    )

    model = SARIMAX(
        model_series,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    result = fit.get_forecast(steps=periods)
    forecast = result.predicted_mean
    conf = result.conf_int(alpha=0.05)

    future_dates = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )
    out = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi": np.asarray(forecast),
        "Batas Bawah 95%": np.asarray(conf.iloc[:, 0]),
        "Batas Atas 95%": np.asarray(conf.iloc[:, 1]),
    })
    return out


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown(
    """
    <div class='brand'>
        <div class='brand-title'>📊 SARIMA Analytics</div>
        <div class='brand-subtitle'>Waterpark Sumenep<br>Forecasting Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "NAVIGASI",
    [
        "Beranda",
        "Data Historis",
        "Hasil SARIMA",
        "Windowing",
        "Forecast Masa Depan",
        "Evaluasi Model",
        "Konfigurasi Model",
    ],
)

st.sidebar.divider()
st.sidebar.markdown(
    "<span class='mini-badge'>MODEL • SARIMA</span>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Data telah melalui preprocessing dan interpolasi linear sesuai notebook penelitian.")

# =========================================================
# BERANDA
# =========================================================
if menu == "Beranda":
    st.markdown(
        """
        <div class='hero'>
            <div class='hero-kicker'>MBKM Skema Riset • Forecasting Time Series</div>
            <div class='hero-title'>Peramalan Kunjungan Wisatawan<br>Waterpark Sumenep</div>
            <div class='hero-text'>
                Dashboard interaktif untuk memvisualisasikan data historis,
                hasil training dan testing SARIMA, skenario windowing 6 → 1 dan
                12 → 1, evaluasi model, serta forecast masa depan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        add_kpi("Jumlah Data", fmt_num(len(historis)), "Observasi bulanan")
    with c2:
        add_kpi("Training", fmt_num(len(training)), "70% data")
    with c3:
        add_kpi("Testing", fmt_num(len(testing)), "30% data")
    with c4:
        add_kpi(
            "Periode",
            f"{historis['Tanggal'].min().year}–{historis['Tanggal'].max().year}",
            "Rentang data historis",
        )

    st.write("")
    st.markdown("<div class='section-title'>Gambaran Data</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Pergerakan jumlah kunjungan wisatawan berdasarkan data yang digunakan dalam penelitian.</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historis["Tanggal"],
        y=historis["Jumlah_Kunjungan"],
        mode="lines+markers",
        name="Jumlah Kunjungan",
        line=dict(width=3, color="#1d5b97"),
        marker=dict(size=6, color="#1d5b97"),
    ))
    fig = chart_layout(fig, height=450)
    fig.update_yaxes(title="Jumlah Kunjungan")
    fig.update_xaxes(title="Tanggal")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Ringkasan Penelitian</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div class='card'><b>🔎 Preprocessing</b><br><span class='metric-explain'>Data disiapkan sebagai deret waktu bulanan dan nilai nol ditangani dengan interpolasi linear.</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div class='card'><b>🧩 Model</b><br><span class='metric-explain'>Model menggunakan struktur SARIMA untuk memodelkan hubungan temporal pada data kunjungan.</span></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div class='card'><b>📏 Evaluasi</b><br><span class='metric-explain'>Performa ditampilkan melalui MAE, RMSE, MAPE, dan R² pada hasil yang telah dihitung.</span></div>",
            unsafe_allow_html=True,
        )

# =========================================================
# DATA HISTORIS
# =========================================================
elif menu == "Data Historis":
    st.title("Data Historis")
    st.caption("Eksplorasi deret waktu kunjungan wisatawan yang digunakan sebagai dasar peramalan.")

    min_date = historis["Tanggal"].min().date()
    max_date = historis["Tanggal"].max().date()
    date_range = st.date_input("Rentang tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = historis[
            (historis["Tanggal"].dt.date >= date_range[0]) &
            (historis["Tanggal"].dt.date <= date_range[1])
        ].copy()
    else:
        filtered = historis.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        add_kpi("Observasi", fmt_num(len(filtered)), "Data pada rentang terpilih")
    with c2:
        add_kpi("Minimum", fmt_num(filtered["Jumlah_Kunjungan"].min(), 0), "Jumlah kunjungan")
    with c3:
        add_kpi("Maksimum", fmt_num(filtered["Jumlah_Kunjungan"].max(), 0), "Jumlah kunjungan")

    st.markdown("<div class='section-title'>Visualisasi Historis</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["Tanggal"], y=filtered["Jumlah_Kunjungan"],
        mode="lines+markers", name="Aktual",
        line=dict(width=3, color="#1d5b97"),
        marker=dict(size=5, color="#1d5b97"),
    ))
    fig = chart_layout(fig, height=430)
    fig.update_yaxes(title="Jumlah Kunjungan")
    fig.update_xaxes(title="Tanggal")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat tabel data"):
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Data Historis",
            filtered.to_csv(index=False).encode("utf-8"),
            "data_historis.csv",
            "text/csv",
        )

# =========================================================
# HASIL SARIMA
# =========================================================
elif menu == "Hasil SARIMA":
    st.title("Hasil Peramalan SARIMA")
    st.caption("Perbandingan nilai aktual dan prediksi pada data training serta testing.")

    mode = st.radio("Dataset", ["Training", "Testing"], index=1, horizontal=True)
    result_df = training if mode == "Training" else testing

    c1, c2, c3 = st.columns(3)
    with c1:
        add_kpi("Observasi", fmt_num(len(result_df)), mode)
    with c2:
        add_kpi("Aktual Rata-rata", fmt_num(result_df["Aktual"].mean(), 0), "Kunjungan")
    with c3:
        add_kpi("Prediksi Rata-rata", fmt_num(result_df["Prediksi"].mean(), 0), "Kunjungan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df["Tanggal"], y=result_df["Aktual"],
        mode="lines+markers", name="Aktual",
        line=dict(width=3, color="#1d5b97"), marker=dict(size=5, color="#1d5b97")
    ))
    fig.add_trace(go.Scatter(
        x=result_df["Tanggal"], y=result_df["Prediksi"],
        mode="lines+markers", name="Prediksi",
        line=dict(width=2.5, dash="dash", color="#f28e2b"), marker=dict(size=5, color="#f28e2b")
    ))
    fig = chart_layout(fig, f"Aktual vs Prediksi — {mode}", height=470)
    fig.update_yaxes(title="Jumlah Kunjungan")
    fig.update_xaxes(title="Tanggal")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Tabel Prediksi</div>", unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Hasil",
        result_df.to_csv(index=False).encode("utf-8"),
        f"hasil_{mode.lower()}.csv",
        "text/csv",
    )

# =========================================================
# WINDOWING
# =========================================================
elif menu == "Windowing":
    st.title("Evaluasi Windowing")
    st.caption("Perbandingan skenario input 6 bulan → 1 keluaran dan 12 bulan → 1 keluaran.")

    window_choice = st.radio(
        "Skenario",
        ["Window 6 → 1", "Window 12 → 1"],
        index=0,
        horizontal=True,
    )
    wdf = window6 if window_choice == "Window 6 → 1" else window12

    c1, c2, c3 = st.columns(3)
    with c1:
        add_kpi("Observasi", fmt_num(len(wdf)), window_choice)
    with c2:
        add_kpi("Aktual Rata-rata", fmt_num(wdf["Aktual"].mean(), 0), "Kunjungan")
    with c3:
        add_kpi("Prediksi Rata-rata", fmt_num(wdf["Prediksi"].mean(), 0), "Kunjungan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wdf["Tanggal"], y=wdf["Aktual"], mode="lines+markers",
        name="Aktual", line=dict(width=3, color="#1d5b97"), marker=dict(size=5, color="#1d5b97")
    ))
    fig.add_trace(go.Scatter(
        x=wdf["Tanggal"], y=wdf["Prediksi"], mode="lines+markers",
        name="Prediksi", line=dict(width=2.5, dash="dash", color="#f28e2b"), marker=dict(size=5, color="#f28e2b")
    ))
    fig = chart_layout(fig, window_choice, height=460)
    fig.update_yaxes(title="Jumlah Kunjungan")
    fig.update_xaxes(title="Tanggal")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat detail data windowing"):
        st.dataframe(wdf, use_container_width=True, hide_index=True)

# =========================================================
# FORECAST MASA DEPAN
# =========================================================
elif menu == "Forecast Masa Depan":
    st.title("Forecast Masa Depan")
    st.caption("Forecast interaktif untuk periode setelah observasi historis terakhir.")

    (p, d, q), (P, D, Q), s = get_model_params()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        add_kpi("AR (p)", str(p), "Komponen autoregresif")
    with c2:
        add_kpi("I (d)", str(d), "Differencing")
    with c3:
        add_kpi("MA (q)", str(q), "Moving average")
    with c4:
        add_kpi("Musiman (s)", str(s), "Periode bulanan")

    st.markdown("<div class='card'><b>Model:</b> SARIMA ({},{},{}) × ({},{},{},{})</div>".format(p,d,q,P,D,Q,s), unsafe_allow_html=True)

    periods = st.slider(
        "Jumlah periode forecast",
        min_value=1,
        max_value=24,
        value=12,
        help="Jumlah bulan yang diprediksi setelah periode historis terakhir.",
    )

    if st.button("🚀 Jalankan Forecast", type="primary", use_container_width=True):
        series = historis[["Tanggal", "Jumlah_Kunjungan"]].dropna().copy()
        series = series.sort_values("Tanggal").set_index("Tanggal")["Jumlah_Kunjungan"].astype(float)

        with st.spinner("Memasang model SARIMA dan menghitung forecast..."):
            try:
                result = build_forecast(series, periods)
            except Exception as exc:
                st.error(f"Forecast gagal dihitung: {exc}")
                st.stop()

        st.success(f"Forecast {periods} bulan berhasil dibuat.")

        st.markdown("<div class='section-title'>Hasil Forecast</div>", unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name="Historis",
            line=dict(width=3, color="#1d5b97")
        ))
        fig.add_trace(go.Scatter(
            x=result["Tanggal"], y=result["Prediksi"],
            mode="lines+markers", name="Forecast",
            line=dict(width=3, color="#f28e2b"), marker=dict(size=6, color="#f28e2b")
        ))
        fig.add_trace(go.Scatter(
            x=list(result["Tanggal"]) + list(result["Tanggal"])[::-1],
            y=list(result["Batas Atas 95%"]) + list(result["Batas Bawah 95%"][::-1]),
            fill="toself",
            fillcolor="rgba(242,142,43,.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Interval 95%",
        ))
        fig = chart_layout(fig, "Historis dan Forecast Masa Depan", height=500)
        fig.update_yaxes(title="Jumlah Kunjungan")
        fig.update_xaxes(title="Tanggal")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Forecast",
            result.to_csv(index=False).encode("utf-8"),
            "forecast_masa_depan.csv",
            "text/csv",
        )

# =========================================================
# EVALUASI MODEL
# =========================================================
elif menu == "Evaluasi Model":
    st.title("Evaluasi Model")
    st.caption("Metrik evaluasi yang tersedia dari hasil perhitungan pada notebook penelitian.")

    # Normalisasi nama kolom agar tampilan tetap robust terhadap variasi CSV.
    dataset_col = find_column(metrics, ["Dataset", "Skenario", "Model"])
    if dataset_col is None:
        dataset_col = metrics.columns[0]

    mae_col = find_column(metrics, ["MAE"])
    rmse_col = find_column(metrics, ["RMSE"])
    mape_col = find_column(metrics, ["MAPE (%)", "MAPE"])
    r2_col = find_column(metrics, ["R²", "R2", "R^2"])

    display_metrics = metrics.copy()
    rename_map = {}
    if dataset_col:
        rename_map[dataset_col] = "Skenario"
    if mae_col:
        rename_map[mae_col] = "MAE"
    if rmse_col:
        rename_map[rmse_col] = "RMSE"
    if mape_col:
        rename_map[mape_col] = "MAPE"
    if r2_col:
        rename_map[r2_col] = "R²"
    display_metrics = display_metrics.rename(columns=rename_map)

    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    # Bar chart metrics jika kolom tersedia.
    if all(c in display_metrics.columns for c in ["Skenario", "MAE", "RMSE"]):
        st.markdown("<div class='section-title'>Perbandingan Nilai Error</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for metric_name, color, dash in [
            ("MAE", "#1d5b97", "solid"),
            ("RMSE", "#f28e2b", "dot"),
        ]:
            fig.add_trace(go.Bar(
                x=display_metrics["Skenario"],
                y=pd.to_numeric(display_metrics[metric_name], errors="coerce"),
                name=metric_name,
                marker_color=color,
            ))
        fig.update_layout(
            barmode="group",
            height=420,
            margin=dict(l=18, r=18, t=20, b=18),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#e8edf3", title="Nilai Error"),
            xaxis=dict(showgrid=False, title="Skenario"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Makna Metrik</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><b>MAE</b><br><span class='metric-explain'>Rata-rata besar kesalahan absolut antara nilai aktual dan prediksi.</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><b>RMSE</b><br><span class='metric-explain'>Akar dari rata-rata kuadrat kesalahan; memberi penalti lebih besar pada error yang besar.</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><b>MAPE</b><br><span class='metric-explain'>Persentase rata-rata kesalahan absolut relatif terhadap nilai aktual.</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><b>R²</b><br><span class='metric-explain'>Proporsi variasi data aktual yang dijelaskan oleh prediksi pada evaluasi yang digunakan.</span></div>", unsafe_allow_html=True)

# =========================================================
# KONFIGURASI MODEL
# =========================================================
elif menu == "Konfigurasi Model":
    st.title("Konfigurasi Model")
    st.caption("Parameter dan konfigurasi yang diekspor dari notebook penelitian.")

    st.markdown("<div class='section-title'>Parameter SARIMA</div>", unsafe_allow_html=True)
    (p, d, q), (P, D, Q), s = get_model_params()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        add_kpi("p", str(p), "AR non-musiman")
    with c2:
        add_kpi("d", str(d), "Differencing non-musiman")
    with c3:
        add_kpi("q", str(q), "MA non-musiman")
    with c4:
        add_kpi("s", str(s), "Musiman bulanan")

    st.markdown(
        f"<div class='card'><b>Struktur:</b> SARIMA ({p},{d},{q}) × ({P},{D},{Q},{s})</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>File Konfigurasi dari Notebook</div>", unsafe_allow_html=True)
    st.dataframe(config, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Alur Singkat</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("01", "Data", "Data bulanan"),
        ("02", "Preprocessing", "Interpolasi linear"),
        ("03", "Model", "SARIMA"),
        ("04", "Evaluasi", "MAE • RMSE • MAPE • R²"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div class='card'><span class='mini-badge'>{num}</span><br><br><b>{title}</b><br><span class='metric-explain'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class='footer'>
        <b>SARIMA Forecast Dashboard — Waterpark Sumenep</b><br>
        Hasil visualisasi dan forecasting bersumber dari data serta konfigurasi yang diekspor dari notebook penelitian.
    </div>
    """,
    unsafe_allow_html=True,
)