import pandas as pd
import requests
import numpy as np
import os
import io 


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

    filename = "./data/processed/data_filtered.parquet"

    features = ['lat',
    'lon',
    'minutes_remaining',
    'period',
    'playoffs',
    'shot_distance',
    'shot_made_flag'
    ]
    
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df = df.dropna()
    df = df.dropna(how='any',axis=0)
    df = df.drop_duplicates()
    df = df[features]

    df.to_parquet(filename)

    if not os.path.exists(filename):
        print("Erro: O dataset não foi processado corretamente!")
        return None

    df = pd.read_parquet(filename)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")

    return df

