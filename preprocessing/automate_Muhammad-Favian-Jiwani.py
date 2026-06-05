import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing data...")
    
    # Konfigurasi Path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, '../diabetes_raw/diabetes_binary_health_indicators_BRFSS2015.csv')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'diabetes_preprocessing')
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'cleaned_diabetes_data.csv')
    
    # Validasi keberadaan file raw
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {RAW_DATA_PATH}")
    
    # 1. Memuat Data
    print("Memuat dataset raw...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. Menghapus Duplikat
    print("Menghapus data duplikat...")
    df_cleaned = df.drop_duplicates()
    
    # 3. Normalisasi Fitur (MinMaxScaler)
    print("Melakukan normalisasi pada fitur BMI, MentHlth, dan PhysHlth...")
    scaler = MinMaxScaler()
    columns_to_scale = ['BMI', 'MentHlth', 'PhysHlth']
    df_cleaned[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])
    
    # 4. Menyimpan Data Bersih
    print("Menyimpan dataset hasil preprocessing...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Preprocessing sukses! File disimpan di: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_preprocessing()