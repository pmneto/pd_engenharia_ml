import pandas as pd
import mlflow
import os
from sklearn.metrics import log_loss, f1_score
import joblib
from sklearn.metrics import log_loss, f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def salvar_metricas_producao(metricas: dict, output_path: str = "data/07_model_output/metricas_producao.csv"):
    """
    Salva métricas de avaliação do modelo na base de produção. Se o arquivo já existir, apenas adiciona nova linha.
    """
    df_novo = pd.DataFrame([metricas])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        df_existente = pd.read_csv(output_path)
        df_novo = pd.concat([df_existente, df_novo], ignore_index=True)

    df_novo.to_csv(output_path, index=False)
    print(f"[INFO] Métricas da aplicação salvas em: {output_path}")


def plot_previsoes_producao(df, output_path="data/08_reporting/prediction_map_prod.png"):
    """
    Gera um gráfico com a imagem da quadra ao fundo e predições do modelo aplicadas na base de produção.
    """
    if "prediction_label" not in df.columns:
        print("[ERRO] Coluna 'prediction_label' não encontrada no DataFrame de produção.")
        return

    try:
        court_img = mpimg.imread("data/01_raw/nba_court.png")
    except FileNotFoundError:
        print("[ERRO] Imagem da quadra não encontrada.")
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(court_img, extent=[-118.5, -118.05, 33.3, 34.2], aspect='auto', zorder=0)

    cores = {0: "red", 1: "green"}
    marcadores = {0: "x", 1: "o"}
    labels = {0: "Errou (0)", 1: "Acertou (1)"}

    for classe in df["prediction_label"].unique():
        subset = df[df["prediction_label"] == classe]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            c=cores[classe],
            marker=marcadores[classe],
            label=labels[classe],
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
            s=50,
            zorder=1,
        )

    plt.title("Previsões do Modelo na Base de Produção")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Previsão")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    mlflow.log_artifact(output_path, artifact_path="plots")
    print(f"[INFO] Gráfico salvo em: {output_path}")



def aplicar_modelo_prod(model_path, df_prod):
    with mlflow.start_run(run_name="PipelineAplicacao"):
        print("[INFO] Rodando pipeline de aplicação...")
        
        cols = [
            'action_type',
            'combined_shot_type',
            'game_event_id',
            'game_id',
            'loc_x',
            'loc_y',
            'season',
            'seconds_remaining',
            'shot_type',
            'shot_zone_area',
            'shot_zone_basic',
            'shot_zone_range',
            'team_id',
            'team_name',
            'game_date',
            'matchup',
            'opponent',
            'shot_id',
            'shot_made_flag'
        ]

        if "shot_made_flag" in df_prod.columns:
             df_prod = df_prod.dropna(subset=cols)


        X_prod = df_prod.drop(columns=cols, errors="ignore")

        model = model_path

        y_pred_proba = model.predict_proba(X_prod)[:, 1]
        y_pred = model.predict(X_prod)

        df_prod["prediction_label"] = y_pred
        df_prod["prediction_proba"] = y_pred_proba

        if "shot_made_flag" in df_prod.columns:
            y_true = df_prod["shot_made_flag"]
            f1 = f1_score(y_true, y_pred)
            ll = log_loss(y_true, y_pred_proba)

            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("log_loss", ll)
            print(f"[INFO] F1: {f1:.4f} | Log Loss: {ll:.4f}")
        else:
            print("[WARN] Coluna shot_made_flag não está presente — sem métricas.")


        metricas = {
            "dataset": "prod",
            "modelo": "MelhorModeloKobe",
            "run_id": mlflow.active_run().info.run_id,
            "log_loss": log_loss(df_prod["shot_made_flag"], df_prod["prediction_proba"]),
            "f1_score": f1_score(df_prod["shot_made_flag"], df_prod["prediction_label"]),
            "accuracy": accuracy_score(df_prod["shot_made_flag"], df_prod["prediction_label"]),
            "precision": precision_score(df_prod["shot_made_flag"], df_prod["prediction_label"]),
            "recall": recall_score(df_prod["shot_made_flag"], df_prod["prediction_label"]),
            "timestamp": datetime.now().isoformat()
        }

        salvar_metricas_producao(metricas)
        plot_previsoes_producao(df_prod)

        os.makedirs("data/07_model_output", exist_ok=True)
        output_path = "data/07_model_output/predictions_prod.parquet"
        df_prod.to_parquet(output_path, index=False)
        mlflow.log_artifact(output_path, artifact_path="predicoes")

       

        return df_prod
