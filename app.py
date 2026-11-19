
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HeartGuard",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# 2. MENGHUBUNGKAN STREAMLIT DENGAN DJANGO
# =====================================================

sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings"
)

import django
django.setup()

from django.db import close_old_connections
from prediction.models import PredictionHistory

# =====================================================
# 3. MEMUAT MODEL MACHINE LEARNING
# =====================================================

MODEL_PATH = BASE_DIR / "prediction" / "heart_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =====================================================
# 4. MEMUAT CSS
# =====================================================

CSS_PATH = BASE_DIR / "static" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        css = file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )

# =====================================================
# 5. KONFIGURASI GEMINI
# =====================================================

from google import genai

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None

# =====================================================
# 6. NAVIGASI APLIKASI
# =====================================================

st.sidebar.title("❤️ HeartGuard")

st.sidebar.write("Sistem Prediksi Penyakit Jantung")

menu = st.sidebar.radio(
    "Menu Utama",
    [
        "Dashboard",
        "Prediksi Penyakit Jantung",
        "Riwayat Prediksi",
        "HeartGuard AI"
    ]
)

# =====================================================
# 7. DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.title("❤️ HeartGuard")
    st.subheader("Dashboard Kesehatan Jantung")

    st.write(
        "Selamat datang di HeartGuard, aplikasi "
        "prediksi risiko penyakit jantung menggunakan "
        "algoritma Decision Tree."
    )

    st.divider()

    close_old_connections()

    try:
        total_prediction = PredictionHistory.objects.count()
    finally:
        close_old_connections()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Prediksi",
            total_prediction
        )

    with col2:
        st.metric(
            "Akurasi Model",
            "85,22%"
        )

    st.info(
        "HeartGuard membantu memberikan gambaran "
        "awal mengenai risiko penyakit jantung "
        "berdasarkan data yang dimasukkan."
    )

    st.warning(
        "Hasil prediksi bukan diagnosis medis. "
        "Silakan konsultasikan kondisi kesehatan "
        "kepada tenaga medis."
    )

# =====================================================
# 8. PREDIKSI PENYAKIT JANTUNG
# =====================================================

elif menu == "Prediksi Penyakit Jantung":

    st.title("❤️ Prediksi Penyakit Jantung")

    st.write(
        "Silakan masukkan data pemeriksaan "
        "untuk melakukan prediksi."
    )

    with st.form("prediction_form"):

        st.subheader("Data Pasien")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Usia",
                min_value=1,
                max_value=120,
                value=30
            )

            sex = st.selectbox(
                "Jenis Kelamin",
                ["M", "F"],
                format_func=lambda x: (
                    "Laki-laki" if x == "M" else "Perempuan"
                )
            )

            chest_pain_type = st.selectbox(
                "Jenis Nyeri Dada",
                ["TA", "ATA", "NAP", "ASY"],
                format_func=lambda x: {
                    "TA": "Typical Angina",
                    "ATA": "Atypical Angina",
                    "NAP": "Non-Anginal Pain",
                    "ASY": "Asymptomatic"
                }[x]
            )

            resting_bp = st.number_input(
                "Tekanan Darah Istirahat (mmHg)",
                min_value=0,
                max_value=300,
                value=120
            )

            cholesterol = st.number_input(
                "Kolesterol (mg/dl)",
                min_value=0,
                max_value=1000,
                value=200
            )

            fasting_bs = st.selectbox(
                "Gula Darah Puasa",
                [0, 1],
                format_func=lambda x: (
                    "Normal (≤ 120 mg/dl)"
                    if x == 0
                    else "Tinggi (> 120 mg/dl)"
                )
            )

        with col2:

            resting_ecg = st.selectbox(
                "Hasil Resting ECG",
                ["Normal", "ST", "LVH"],
                format_func=lambda x: {
                    "Normal": "Normal",
                    "ST": "ST-T Wave Abnormality",
                    "LVH": "Left Ventricular Hypertrophy"
                }[x]
            )

            max_heart_rate = st.number_input(
                "Detak Jantung Maksimum",
                min_value=0,
                max_value=300,
                value=150
            )

            exercise_angina = st.selectbox(
                "Angina Saat Olahraga",
                ["N", "Y"],
                format_func=lambda x: (
                    "Tidak" if x == "N" else "Ya"
                )
            )

            oldpeak = st.number_input(
                "Oldpeak",
                min_value=-5.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )

            st_slope = st.selectbox(
                "ST Slope",
                ["Up", "Flat", "Down"],
                format_func=lambda x: {
                    "Up": "Upsloping",
                    "Flat": "Flat",
                    "Down": "Downsloping"
                }[x]
            )

        submitted = st.form_submit_button(
            "🔍 Prediksi Sekarang",
            use_container_width=True
        )

    # Memproses prediksi
    if submitted:

        input_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain_type,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_heart_rate,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        try:

            result = int(model.predict(input_data)[0])

            # Menyimpan hasil ke SQLite Django
            close_old_connections()

            try:
                PredictionHistory.objects.create(
                    age=int(age),
                    sex=sex,
                    chest_pain_type=chest_pain_type,
                    resting_bp=float(resting_bp),
                    cholesterol=float(cholesterol),
                    fasting_bs=int(fasting_bs),
                    resting_ecg=resting_ecg,
                    max_heart_rate=int(max_heart_rate),
                    exercise_angina=exercise_angina,
                    oldpeak=float(oldpeak),
                    st_slope=st_slope,
                    prediction_result=result
                )
            finally:
                close_old_connections()

            st.divider()

            st.subheader("Hasil Prediksi")

            if result == 1:

                st.error(
                    "Hasil model: Terindikasi memiliki "
                    "risiko penyakit jantung."
                )

            else:

                st.success(
                    "Hasil model: Tidak terindikasi "
                    "memiliki risiko penyakit jantung."
                )

            st.info(
                "Hasil prediksi berhasil disimpan "
                "ke database SQLite Django."
            )

            st.caption(
                "Hasil ini merupakan prediksi model, "
                "bukan diagnosis medis."
            )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat melakukan prediksi: {e}"
            )

# =====================================================
# 9. RIWAYAT PREDIKSI
# =====================================================

elif menu == "Riwayat Prediksi":

    st.title("📋 Riwayat Prediksi")

    st.write(
        "Berikut merupakan data prediksi "
        "yang tersimpan dalam database Django."
    )

    close_old_connections()

    try:
        histories = list(
            PredictionHistory.objects.all().order_by(
                "-prediction_date"
            )
        )
    finally:
        close_old_connections()

    if histories:

        history_data = []

        for item in histories:

            history_data.append({
                "ID": item.id,
                "Usia": item.age,
                "Jenis Kelamin": item.sex,
                "Nyeri Dada": item.chest_pain_type,
                "Tekanan Darah": item.resting_bp,
                "Kolesterol": item.cholesterol,
                "Hasil Prediksi": (
                    "Berisiko"
                    if item.prediction_result == 1
                    else "Tidak Berisiko"
                ),
                "Tanggal Prediksi": item.prediction_date
            })

        history_df = pd.DataFrame(history_data)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Total Riwayat Prediksi",
            len(histories)
        )

    else:

        st.info(
            "Belum ada riwayat prediksi. "
            "Silakan lakukan prediksi terlebih dahulu."
        )

# =====================================================
# 10. CHATBOT HEARTGUARD AI
# =====================================================

elif menu == "HeartGuard AI":

    st.title("💬 HeartGuard AI")

    st.write(
        "Asisten AI untuk membantu menjawab "
        "pertanyaan seputar kesehatan jantung."
    )

    if not client:

        st.warning(
            "Gemini belum terhubung. "
            "Periksa konfigurasi GEMINI_API_KEY "
            "di Streamlit secrets."
        )

    else:

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(
            "Tulis pertanyaan tentang kesehatan jantung..."
        )

        if prompt:

            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            system_instruction = """
            Kamu adalah HeartGuard AI Assistant.

            Bantu pengguna memahami kesehatan jantung.
            Gunakan bahasa Indonesia yang sederhana,
            ramah, dan mudah dipahami.

            Jangan memberikan diagnosis medis.
            Sarankan konsultasi dengan tenaga medis.

            Jika pengguna mengalami nyeri dada berat,
            sesak napas, atau pingsan, sarankan segera
            mencari pertolongan medis darurat.
            """

            conversation = system_instruction + "\n\n"

            for message in st.session_state.messages:
                conversation += (
                    message["role"] + ": "
                    + message["content"] + "\n"
                )

            try:

                with st.chat_message("assistant"):

                    with st.spinner(
                        "HeartGuard AI sedang berpikir..."
                    ):

                        response = client.models.generate_content(
                            model="gemini-3.5-flash-lite",
                            contents=conversation
                        )

                        answer = response.text

                    st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"Gagal menghubungi Gemini: {e}"
                )
