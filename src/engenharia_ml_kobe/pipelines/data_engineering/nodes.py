import pandas as pd
import requests
import os
import mlflow


def download_and_load_data():
    """Baixa o dataset de desenvolvimento e retorna um DataFrame."""
    DATASET_URL = "https://github.com/tciodaro/eng_ml/raw/main/data/dataset_kobe_dev.parquet"
    filename = "data/raw/dataset_kobe_dev.parquet"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    response = requests.get(DATASET_URL)
    with open(filename, "wb") as f:
        f.write(response.content)

    df = pd.read_parquet(filename)
    print(f"[DEV] Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df


def preprocess_data(df: pd.DataFrame, is_prod=False) -> pd.DataFrame:
    """Pré-processa os dados (dev ou prod) e salva resultado no caminho correto."""
    mlflow.set_experiment("engenharia_ml")
    run_name = "PreparacaoDados"

    with mlflow.start_run(run_name=run_name):
        filename = (
           "data/processed/data_filtered.parquet"
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        features = ['lat', 'lng', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']

        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        if 'lon' in df.columns:
            df = df.rename(columns={'lon': 'lng'})

        df = df.dropna().drop_duplicates()
        df = df[features]

        df.to_parquet(filename)
        mlflow.log_artifact(filename)

        print(f"[{'PROD' if is_prod else 'DEV'}] Dataset salvo com {df.shape[0]} linhas e {df.shape[1]} colunas")
        return df


def download_and_load_data_prod():
    """Baixa o dataset de produção, aplica pré-processamento e retorna um DataFrame."""
    DATASET_URL_PROD = "https://github.com/tciodaro/eng_ml/raw/main/data/dataset_kobe_prod.parquet"
    filename = "data/raw/dataset_kobe_prod.parquet"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    response = requests.get(DATASET_URL_PROD)
    with open(filename, "wb") as f:
        f.write(response.content)

    df = pd.read_parquet(filename)
    print(f"[PROD] Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
    return preprocess_data(df, is_prod=True)
