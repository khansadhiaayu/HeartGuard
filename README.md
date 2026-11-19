# HeartGuard - Sistem Prediksi Risiko Penyakit Jantung

**HeartGuard** adalah aplikasi web berbasis Django yang mengintegrasikan model *Machine Learning* untuk memprediksi risiko penyakit jantung berdasarkan data klinis pasien (`heart.csv`), dilengkapi dengan visualisasi grafik data serta penjelasan hasil prediksi berbasis AI menggunakan Google Gemini API.

## Fitur Utama
* **Dashboard Visualisasi Data:** Menampilkan ringkasan statistik dan grafik eksplorasi dari dataset kesehatan jantung.
* **Prediksi Risiko Jantung:** Formulir input parameter klinis pasien untuk memprediksi tingkat risiko penyakit jantung menggunakan model *Machine Learning* terlatih.
* **Penjelasan Berbasis AI (Gemini):** Memberikan interpretasi dan saran edukatif secara otomatis berdasarkan hasil prediksi pasien.
* **Riwayat Prediksi:** Menyimpan dan menampilkan daftar hasil prediksi yang telah dilakukan sebelumnya.

## Teknologi yang Digunakan
* **Backend:** Python, Django
* **Machine Learning & Data:** Scikit-learn, Pandas, NumPy, Joblib
* **AI Integration:** Google GenAI SDK (`gemini-2.5-flash`)
* **Database:** SQLite3
