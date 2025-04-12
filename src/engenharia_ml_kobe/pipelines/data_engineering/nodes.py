import pandas as pd
import requests
import io


def download_and_load_data() -> pd.DataFrame:
    """Baixa o dataset de desenvolvimento e retorna como DataFrame."""
    DATASET_URL = "https://github.com/tciodaro/eng_ml/raw/main/data/dataset_kobe_dev.parquet"
    response = requests.get(DATASET_URL)
    df = pd.read_parquet(io.BytesIO(response.content))

    print(f"[download_and_load_data] Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa os dados: limpeza, padronização e filtragem."""
    features = [
        "lat", "lon", "minutes_remaining", "period",
        "playoffs", "shot_distance", "shot_made_flag"
    ]

    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df = df.dropna().drop_duplicates()
    df = df[features]

    print(f"[preprocess_data] Dataset pré-processado com {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df


def download_and_load_data_prod() -> pd.DataFrame:
    """Baixa o dataset de produção e retorna como DataFrame."""
    DATASET_URL_PROD = "https://github.com/tciodaro/eng_ml/raw/main/data/dataset_kobe_prod.parquet"
    response = requests.get(DATASET_URL_PROD)
    df = pd.read_parquet(io.BytesIO(response.content))

    print(f"[download_and_load_data_prod] Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df
