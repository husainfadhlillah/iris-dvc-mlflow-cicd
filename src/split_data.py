import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml


def split_data():
    """Membagi data mentah menjadi set latih dan uji."""
    print("Memulai proses split_data...")
    # Buat folder jika belum ada
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Baca parameter
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    test_size = params['split_data']['test_size']
    random_state = params['split_data']['random_state']

    # Baca data mentah
    df = pd.read_csv("data/raw/iris.csv")

    # Bagi data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['target']
    )

    # Simpan
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Data latih dan uji berhasil disimpan di data/processed/")


if __name__ == "__main__":
    split_data()