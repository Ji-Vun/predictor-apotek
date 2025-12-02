from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


app = Flask(__name__)

DATASET_PATH = "kaggle_hourly.csv"

def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    # Pastikan nama kolom benar
    df['datum'] = pd.to_datetime(df['datum'])

    # list kolom obat
    obat_cols = ["M01AB","M01AE","N02BA","N02BE","N05B","N05C","R03","R06"]

    # fitur datetime
    df["Year"] = df["datum"].dt.year
    df["Month"] = df["datum"].dt.month
    df["Day"] = df["datum"].dt.day
    df["Hour"] = df["datum"].dt.hour

    return df, obat_cols

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict_page")
def predict_page():
    df, obat_cols = load_dataset()
    return render_template("index.html", obat_list=obat_cols)

@app.route("/index")
def index():
    df, obat_cols = load_dataset()
    return render_template("index.html", obat_list=obat_cols)



@app.route("/predict", methods=["POST"])
def predict():
    nama_obat = request.form["nama_obat"]

    df, obat_cols = load_dataset()

    obat = request.form["obat"]
    start = request.form["start"]
    end = request.form["end"]

    # ubah ke datetime
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if start > end:
        return render_template(
            "index.html",
            obat_list=obat_cols,
            error="❌ Tanggal mulai tidak boleh lebih besar dari tanggal akhir!"
        )

    # ====== SIAPKAN MODEL RANDOM FOREST PER JAM ======
    X = df[["Year","Month","Day","Hour"]]
    y = df[obat]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # ====== BUAT RANGE TANGGAL BARU PER JAM ======
    future_hours = pd.date_range(start=start, end=end, freq="H")

    future_df = pd.DataFrame({
        "Year": future_hours.year,
        "Month": future_hours.month,
        "Day": future_hours.day,
        "Hour": future_hours.hour
    })

    # ====== PREDIKSI PER JAM ======
    pred_per_hour = model.predict(future_df)

    # Dataframe agregasi
    result = pd.DataFrame({
        "datum": future_hours,
        "prediksi": pred_per_hour
    })

    # ====== AGREGASI PER HARI ======
    result["tanggal"] = result["datum"].dt.date
    daily = result.groupby("tanggal")["prediksi"].sum().reset_index()

    daily["prediksi"] = daily["prediksi"].round(2)

    # --- HITUNG REKOMENDASI STOK ---
    max_pred = daily["prediksi"].max()          # permintaan tertinggi per hari
    safety_stock = max_pred * 0.20              # 20% cadangan stok
    rekomendasi_stok = round(max_pred + safety_stock, 2)

    # ==== BUAT GRAFIK DALAM BENTUK BASE64 UNTUK HTML ====
    plt.figure(figsize=(10,5))
    plt.plot(daily["tanggal"], daily["prediksi"])
    plt.xlabel("Tanggal")
    plt.ylabel("Prediksi Penggunaan")
    plt.title(f"Grafik Prediksi Penggunaan {obat}")
    plt.xticks(rotation=45)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    grafik_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return render_template(
    "result.html",
    obat=obat,
    nama_obat=nama_obat,
    start=start.date(),
    end=end.date(),
    table=daily.values.tolist(),
    grafik=grafik_base64,
    rekomendasi_stok=rekomendasi_stok
)





if __name__ == "__main__":
    app.run(debug=True)
