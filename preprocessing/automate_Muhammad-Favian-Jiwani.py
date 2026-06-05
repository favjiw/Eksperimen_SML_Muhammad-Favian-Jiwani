import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing data...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(
        BASE_DIR, "..", "diabetes_raw",
        "diabetes_binary_health_indicators_BRFSS2015.csv",   
    )
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "diabetes_preprocessing")
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "cleaned_diabetes_data.csv")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {RAW_DATA_PATH}")

    print("Memuat dataset raw...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Menghapus data duplikat...")
    df_cleaned = df.drop_duplicates().reset_index(drop=True).copy()

    print("Normalisasi fitur BMI, MentHlth, PhysHlth...")
    scaler = MinMaxScaler()
    columns_to_scale = ["BMI", "MentHlth", "PhysHlth"]
    df_cleaned[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

    print("Menyimpan dataset hasil preprocessing...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing sukses! File disimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_preprocessing()