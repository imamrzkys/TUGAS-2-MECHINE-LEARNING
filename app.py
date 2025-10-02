from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model_heart_rate.pkl")

# Nilai R2 dan RMSE (ganti dengan nilai aktual dari model Anda)
R2_SCORE = 0.95  # Contoh nilai, sesuaikan dengan hasil training
RMSE_SCORE = 5.2  # Contoh nilai, sesuaikan dengan hasil training

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    duration_input = None
    
    if request.method == "POST":
        try:
            # Ambil input durasi dari form
            duration_input = float(request.form.get("duration"))
            
            # Prediksi menggunakan model
            # Model di-train dengan DataFrame, jadi kita perlu menggunakan DataFrame dengan nama kolom
            input_data = pd.DataFrame([[duration_input]], columns=['Duration'])
            prediction = model.predict(input_data)[0]
            prediction = round(prediction, 2)
            
        except (ValueError, TypeError):
            prediction = "Error: Masukkan angka yang valid"
    
    return render_template(
        "index.html",
        prediction=prediction,
        duration=duration_input,
        r2=R2_SCORE,
        rmse=RMSE_SCORE
    )

if __name__ == "__main__":
    app.run(debug=True)
