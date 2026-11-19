
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Lokasi dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "heart.csv"

# Membaca dataset
data = pd.read_csv(DATASET_PATH)

# Memisahkan fitur dan target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Menentukan kolom kategori dan numerik
categorical_features = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

numerical_features = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

# Mengolah data kategori
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Membuat model Decision Tree
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=3,
                random_state=42
            )
        )
    ]
)

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Melatih model
model.fit(X_train, y_train)

# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menghitung evaluasi model
accuracy = accuracy_score(y_test, y_pred)

print("=== HASIL PELATIHAN MODEL HEARTGUARD ===")
print("Jumlah data:", len(data))
print("Data training:", len(X_train))
print("Data testing:", len(X_test))
print("Akurasi model:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Menyimpan model
MODEL_PATH = Path(__file__).resolve().parent / "heart_model.joblib"

joblib.dump(model, MODEL_PATH)

print("\nModel berhasil disimpan di:", MODEL_PATH)
