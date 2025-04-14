from pycaret.classification import ClassificationExperiment


import mlflow
import os
from sklearn.metrics import log_loss, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, roc_curve, auc
from mlflow.models.signature import infer_signature

import matplotlib.image as mpimg
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import os
import mlflow
import pandas as pd








def plot_roc_curve(modelo , y_true, y_proba, output_path="data/08_reporting/roc_curve.png"):
    """
    Gera e salva a curva ROC.
    """

    if modelo.lower() == 'l':
        output_path = "data/08_reporting/roc_curve_reg_log.png"
    elif modelo.lower() == 'dt':
        output_path = "data/08_reporting/roc_curve_dt.png"


    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Curva ROC salva em: {output_path}")






def plot_confusion_matrix(modelo , y_true, y_pred, output_path="data/08_reporting/confusion_matrix_reg_log.png"):
    """
    Gera e salva a matriz de confusão.
    """

    if modelo.lower() == 'l':
        output_path = "data/08_reporting/confusion_matrix.png"
    elif modelo.lower() == 'dt':
        output_path = "data/08_reporting/confusion_matrix_dt.png"

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Matriz de confusão salva em: {output_path}")







def salvar_metricas_csv(modelo: str, metricas: dict, output_path: str = "data/07_model_output/metricas_modelo_escolhido.csv"):
    """
    Salva as métricas do modelo em um arquivo CSV. Se o arquivo existir, apenda as métricas.
    """

    if modelo.lower() == 'l':
        output_path = "data/07_model_output/metricas_modelo_escolhido_reg_log.csv"
    elif modelo.lower() == 'dt':
        output_path = "data/07_model_output/metricas_modelo_escolhido_dt.csv"

    df_metricas = pd.DataFrame([metricas])  # Uma linha só
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Verifica se o arquivo já existe
    arquivo_existe = os.path.exists(output_path)

    # Salva no modo append, sem sobrescrever, e adiciona o cabeçalho apenas se for novo
    df_metricas.to_csv(output_path, index=False, mode='a', header=not arquivo_existe)
    print(f"[INFO] Métricas salvas em: {output_path}")







def plot_previsoes_modelo(modelo, df, output_path="data/08_reporting/prediction_map.png"):
    """
    Gera um scatterplot mostrando a previsão do modelo (0 ou 1) com base na posição (lat/lon).
    """
    if "prediction_label" not in df.columns:
        print("[ERRO] Coluna 'prediction_label' não encontrada no DataFrame.")
        return


    if modelo.lower() == 'l':
         output_path: str = "data/08_reporting/prediction_map_reg_log.png"
    elif modelo.lower() == 'dt':
         output_path: str = "data/08_reporting/prediction_map_dt.png"
    

    plt.figure(figsize=(10, 8))
    cores = {0: "red", 1: "green"}
    labels = {0: "Errou (0)", 1: "Acertou (1)"}

    for classe in df["prediction_label"].unique():
        subset = df[df["prediction_label"] == classe]
        plt.scatter(
            subset["lng"],
            subset["lat"],
            c=cores[classe],
            label=labels[classe],
            alpha=0.6,
            s=30
        )

    plt.title("Local dos Arremessos com Previsões do Modelo")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Previsão")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Loga como artefato no MLflow
    mlflow.log_artifact(output_path, artifact_path="plots")
    print(f"[INFO] Gráfico salvo em {output_path} e logado no MLflow.")



def plot_data_filtered(df, output_path="data/08_reporting/prediction_map_raw_data.png"):
    """
    Gera um scatterplot com imagem da quadra ao fundo, com previsões de arremesso (coluna shot_made_flag).
    """

    # Verifica se a imagem existe
    try:
        court_img = mpimg.imread("data/01_raw/nba_court.png")
    except FileNotFoundError:
        print("[ERRO] Imagem da quadra não encontrada! Verifique o caminho.")
        return

    # Verifica se a coluna existe
    if "shot_made_flag" not in df.columns:
        print("[ERRO] Coluna 'shot_made_flag' não encontrada no DataFrame.")
        return

    # Cria a figura com proporção ideal para meia quadra
    fig, ax = plt.subplots(figsize=(16, 9))

    # Desenha a imagem da quadra como fundo
    ax.imshow(court_img, extent=[-118.5, -118.05, 33.3, 34.2], aspect='auto', zorder=0)

    # Define estilo de cada classe
    cores = {0: "red", 1: "green"}
    marcadores = {0: "x", 1: "o"}
    labels = {0: "Errou (0)", 1: "Acertou (1)"}

    # Plota cada classe
    for classe in sorted(df["shot_made_flag"].dropna().unique()):
        subset = df[df["shot_made_flag"] == classe]
        ax.scatter(
            subset["lng"],
            subset["lat"],
            c=cores.get(classe, "gray"),
            marker=marcadores.get(classe, "."),
            label=labels.get(classe, f"Desconhecido ({classe})"),
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
            s=50,
            zorder=1
        )

    ax.set_title("Local dos Arremessos - Dataset Original")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Previsão")
    plt.tight_layout()

    # Garante que o diretório existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    

    # Salva e finaliza
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Gráfico salvo em: {output_path}")




def setup_experiment(data, session_id: int, cv_folds: int) -> ClassificationExperiment:
    exp = ClassificationExperiment()
    exp.setup(
        data=data,
        target="shot_made_flag",
        session_id=session_id,
        train_size=0.8,
        fold=cv_folds,
        fold_shuffle=True,
        html=False,
        log_experiment=False,
        log_plots=False,
        verbose=False
    )
    return exp





def extract_metrics(df):
    if "Label" in df.columns and "Score" in df.columns:
        return df["shot_made_flag"], df["Label"], df["Score"]
    elif "prediction_label" in df.columns and "prediction_score" in df.columns:
        return df["shot_made_flag"], df["prediction_label"], df["prediction_score"]
    else:
        raise ValueError(f"Colunas de predição não encontradas! Colunas disponíveis: {df.columns}")





def train_logistic_model(data, session_id: int, cv_folds: int):

    mlflow.set_experiment("kobe_classificacao")
    with mlflow.start_run(run_name="Treinamento"):
        exp = setup_experiment(data, session_id, cv_folds)

        logistic = exp.create_model("lr", fold=cv_folds)
        tuned_logistic = exp.tune_model(logistic, n_iter=10, fold=cv_folds, optimize="AUC")
        pred = exp.predict_model(tuned_logistic)
        plot_previsoes_modelo('l',pred)



        y_test, y_pred_label, y_pred_proba = extract_metrics(pred)
        mlflow.log_metric("log_loss", log_loss(y_test, y_pred_proba))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_label))

       
        metricas = {
            "model_type": type(tuned_logistic).__name__,
            "log_loss": log_loss(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred_label),
            "precision": precision_score(y_test, y_pred_label, zero_division=0),
            "recall": recall_score(y_test, y_pred_label, zero_division=0),
            "f1_score": f1_score(y_test, y_pred_label),
            "cv_folds": cv_folds,
            "train_rows": len(data),
            "timestamp":datetime.now().isoformat()
        }

        salvar_metricas_csv('l',metricas)

        plot_confusion_matrix('l',y_test, y_pred_label)
        plot_roc_curve('l',y_test, y_pred_proba)

        os.makedirs("data/06_models", exist_ok=True)
        X_input = pred.drop(columns=["prediction_label", "prediction_score", "Label", "Score", "shot_made_flag"], errors="ignore")
        exp.save_model(tuned_logistic, "data/06_models/logistic_model")

       
        mlflow.sklearn.log_model(
            sk_model=tuned_logistic,           # seu modelo treinado
            artifact_path="model",             # o nome do diretório no MLflow
            input_example=X_input.iloc[:1],     # opcional mas recomendado
            signature=mlflow.models.infer_signature(X_input, tuned_logistic.predict(X_input))
        )

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_metric("train_rows", len(data))
        mlflow.log_metric("test_rows", len(y_test))


    return tuned_logistic







def train_best_classifier(data, session_id: int, cv_folds: int):

    mlflow.set_experiment("kobe_classificacao")
    with mlflow.start_run(run_name="Treinamento"):
        exp = setup_experiment(data, session_id, cv_folds)

        best_model = exp.compare_models(exclude=["lr"], fold=cv_folds, n_select=1)
        tuned_best = exp.tune_model(best_model, n_iter=10, fold=cv_folds, optimize="AUC")
        pred = exp.predict_model(tuned_best)
        plot_previsoes_modelo('c',pred)

        y_test, y_pred_label, y_pred_proba = extract_metrics(pred)
        mlflow.log_metric("log_loss", log_loss(y_test, y_pred_proba))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_label))


        metricas = {
            "model_type": type(tuned_best).__name__,
            "log_loss": log_loss(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred_label),
            "precision": precision_score(y_test, y_pred_label, zero_division=0),
            "recall": recall_score(y_test, y_pred_label, zero_division=0),
            "f1_score": f1_score(y_test, y_pred_label),
            "cv_folds": cv_folds,
            "train_rows": len(data),
            "timestamp":datetime.now().isoformat()
        }

        salvar_metricas_csv('c',metricas)

        plot_confusion_matrix('c',y_test, y_pred_label)
        plot_roc_curve('c',y_test, y_pred_proba)

        os.makedirs("data/06_models", exist_ok=True)
        # Salva o modelo treinado
        X_input = pred.drop(columns=["prediction_label", "prediction_score", "Label", "Score", "shot_made_flag"], errors="ignore")
        _, model_path = exp.save_model(tuned_best, "data/06_models/best_classifier")

        # Faz o log do artefato corretamente
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_input.iloc[:1],
            signature=infer_signature(X_input, best_model.predict(X_input))
        )

        mlflow.log_param("model_type", type(tuned_best).__name__)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_metric("train_rows", len(data))
        mlflow.log_metric("test_rows", len(y_test))


    return tuned_best

def train_decision_tree_model(data: pd.DataFrame, session_id: int, cv_folds: int):
    """Treina e registra um modelo de árvore de decisão com PyCaret e MLflow."""


    mlflow.set_experiment("kobe_classificacao")

    with mlflow.start_run(run_name="Treinamento_ArvoreDecisao"):
        # Setup do experimento
        exp = ClassificationExperiment()
        exp.setup(
            data=data,
            target="shot_made_flag",
            session_id=session_id,
            train_size=0.8,
            fold=cv_folds,
            fold_shuffle=True,
            html=False,
            log_experiment=False,
            verbose=False
        )

        # Criação e tuning do modelo de árvore de decisão
        dt_model = exp.create_model("dt", fold=cv_folds)
        tuned_dt = exp.tune_model(dt_model, n_iter=10, fold=cv_folds, optimize="F1")

        # Predição
        pred = exp.predict_model(tuned_dt)

        # Extração de métricas
        y_test = pred["shot_made_flag"]
        y_pred_label = pred["prediction_label"]
        y_pred_proba = pred["prediction_score"]
        plot_previsoes_modelo('dt',pred)
        logloss = log_loss(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred_label)


        metricas = {
            "model_type": type(tuned_dt).__name__,
            "log_loss": log_loss(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred_label),
            "precision": precision_score(y_test, y_pred_label, zero_division=0),
            "recall": recall_score(y_test, y_pred_label, zero_division=0),
            "f1_score": f1_score(y_test, y_pred_label),
            "cv_folds": cv_folds,
            "train_rows": len(data),
            "timestamp":datetime.now().isoformat()
        }

        salvar_metricas_csv('dt',metricas)
        plot_confusion_matrix('dt',y_test, y_pred_label)
        plot_roc_curve('dt',y_test, y_pred_proba)

        # Log de métricas
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_rows", int(len(data) * 0.8))
        mlflow.log_metric("test_rows", int(len(data) * 0.2))

        # Log de parâmetros
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("test_size", 0.2)

        # Salvar modelo
        os.makedirs("data/06_models", exist_ok=True)
        X_input = pred.drop(columns=["prediction_label", "prediction_score", "Label", "Score", "shot_made_flag"], errors="ignore")
        exp.save_model(tuned_dt, "data/06_models/decisionTree")

        # Log do modelo completo
        mlflow.sklearn.log_model(
            sk_model=tuned_dt,
            artifact_path="model",
            input_example=X_input.iloc[:1],
            signature=infer_signature(X_input, tuned_dt.predict(X_input))
        )

        print("[OK] Modelo de árvore de decisão treinado e registrado no MLflow.")
        return tuned_dt
