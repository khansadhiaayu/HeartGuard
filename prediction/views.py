import os
import joblib
import pandas as pd

from google import genai
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from dotenv import load_dotenv
from .models import PredictionHistory

load_dotenv()


# Memuat model Machine Learning
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "prediction" / "heart_model.joblib"

model = joblib.load(MODEL_PATH)


def home(request):
    return render(request, 'prediction/home.html')


def predict(request):
    result = None
    error = None

    if request.method == 'POST':
        try:
            # Mengambil data dari formulir
            input_data = {
                'Age': int(request.POST['age']),
                'Sex': request.POST['sex'],
                'ChestPainType': request.POST['chest_pain_type'],
                'RestingBP': float(request.POST['resting_bp']),
                'Cholesterol': float(request.POST['cholesterol']),
                'FastingBS': int(request.POST['fasting_bs']),
                'RestingECG': request.POST['resting_ecg'],
                'MaxHR': int(request.POST['max_heart_rate']),
                'ExerciseAngina': request.POST['exercise_angina'],
                'Oldpeak': float(request.POST['oldpeak']),
                'ST_Slope': request.POST['st_slope'],
            }

            # Mengubah input menjadi DataFrame
            input_df = pd.DataFrame([input_data])

            # Melakukan prediksi
            prediction = int(model.predict(input_df)[0])

            # Menyimpan hasil ke SQLite
            PredictionHistory.objects.create(
                age=input_data['Age'],
                sex=input_data['Sex'],
                chest_pain_type=input_data['ChestPainType'],
                resting_bp=input_data['RestingBP'],
                cholesterol=input_data['Cholesterol'],
                fasting_bs=input_data['FastingBS'],
                resting_ecg=input_data['RestingECG'],
                max_heart_rate=input_data['MaxHR'],
                exercise_angina=input_data['ExerciseAngina'],
                oldpeak=input_data['Oldpeak'],
                st_slope=input_data['ST_Slope'],
                prediction_result=prediction
            )

            result = prediction

        except (ValueError, KeyError):
            error = "Data yang dimasukkan tidak valid. Periksa kembali formulir."

    return render(request, 'prediction/predict.html', {
        'result': result,
        'error': error
    })

def history(request):
    histories = PredictionHistory.objects.all().order_by('-prediction_date')

    return render(request, 'prediction/history.html', {
        'histories': histories
    })


def dashboard(request):
    # Membaca dataset
    dataset_path = BASE_DIR / "heart.csv"
    df = pd.read_csv(dataset_path)

    # Menghitung ringkasan data
    total_data = len(df)
    tanpa_penyakit = int((df['HeartDisease'] == 0).sum())
    dengan_penyakit = int((df['HeartDisease'] == 1).sum())
    rata_rata_usia = round(df['Age'].mean(), 1)

    # Menghitung jumlah berdasarkan jenis kelamin
    jumlah_laki = int((df['Sex'] == 'M').sum())
    jumlah_perempuan = int((df['Sex'] == 'F').sum())

    context = {
        'total_data': total_data,
        'tanpa_penyakit': tanpa_penyakit,
        'dengan_penyakit': dengan_penyakit,
        'rata_rata_usia': rata_rata_usia,
        'jumlah_laki': jumlah_laki,
        'jumlah_perempuan': jumlah_perempuan,
    }

    return render(request, 'prediction/dashboard.html', context)


@require_POST
def chatbot(request):
    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return JsonResponse({
                "reply": "API Key Gemini belum ditemukan. Periksa file .env kamu."
            }, status=500)

        user_message = request.POST.get("message", "").strip()

        if not user_message:
            return JsonResponse({
                "reply": "Silakan tuliskan pertanyaan terlebih dahulu."
            }, status=400)

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=user_message,
            config={
                "system_instruction": (
                    "Kamu adalah HeartGuard AI, asisten kesehatan jantung. "
                    "Jawab menggunakan bahasa Indonesia yang sederhana "
                    "dan ramah. Berikan informasi umum tentang kesehatan "
                    "jantung. Jangan memberikan diagnosis atau menggantikan "
                    "dokter. Jika pengguna mengalami nyeri dada berat, "
                    "sesak napas, atau pingsan, sarankan segera mencari "
                    "pertolongan medis darurat."
                )
            }
        )

        return JsonResponse({
            "reply": response.text or "Maaf, belum ada jawaban."
        })

    except Exception as e:
        print("Chatbot error:", str(e))

        return JsonResponse({
            "reply": "Maaf, chatbot sedang mengalami kendala. Silakan coba lagi."
        }, status=500)
