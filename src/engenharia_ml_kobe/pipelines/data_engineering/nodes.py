import pandas as pd
import requests
import numpy as np
import os

DATASET_URL = "https://github.com/tciodaro/eng_ml/raw/main/data/dataset_kobe_dev.parquet"

def download_and_load_data():
    """Baixa o dataset e retorna um DataFrame."""
    response = requests.get(DATASET_URL)
    filename = "data/01_raw/dataset_kobe_dev.parquet"

    with open(filename, "wb") as f:
        f.write(response.content)

    if not os.path.exists(filename):
        print("Erro: O dataset não foi baixado corretamente!")
        return None

    df = pd.read_parquet(filename)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove valores nulos e faz pré-processamento inicial."""
    df = df.dropna()
    return df

