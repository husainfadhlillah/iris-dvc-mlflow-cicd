import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

def get_data():
    """Memuat dataset Iris dan menyimpannya sebagai CSV."""
    print("Memulai proses get_data...")
    # Buat folder jika belum ada
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Muat dataset
    iris = load_iris()

    # Konversi ke DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Simpan ke CSV
    df.to_csv("data/raw/iris.csv", index=False)
    print("Dataset mentah berhasil disimpan di data/raw/iris.csv")

if __name__ == "__main__":
    get_data()