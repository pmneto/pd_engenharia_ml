from pycaret.classification import ClassificationExperiment
import mlflow
import os

def train_and_save_models(data, session_id: int, cv_folds: int) -> dict:
    exp = ClassificationExperiment()

    # Setup do experimento sem logging automático
    exp.setup(
        data=data,
        target="shot_made_flag",
        session_id=session_id,
        train_size=0.8,
        fold=cv_folds,
        fold_shuffle=True,
        html=False,
        log_experiment=False,
        experiment_name="kobe_classificacao",
        log_plots=False,
        verbose=False
    )

    mlflow.set_experiment("kobe_classificacao")

    # Inicia experimento manualmente no MLflow
    with mlflow.start_run(run_name="Treinamento"):
        
        # Criação e tuning da Regressão Logística
        logistic = exp.create_model("lr", fold=cv_folds)
        tuned_logistic = exp.tune_model(logistic, n_iter=10, fold=cv_folds, optimize="AUC")

        # Escolha do melhor modelo (exceto a Regressão Logística) e tuning
        best_model = exp.compare_models(exclude=["lr"], fold=cv_folds, n_select=1)
        tuned_best = exp.tune_model(best_model, n_iter=10, fold=cv_folds, optimize="AUC")

        # Identificação do nome do melhor modelo
        model_name = type(tuned_best).__name__
        print(f"[INFO] Melhor modelo escolhido: {model_name}")
        mlflow.log_param("melhor_modelo", model_name)

        # Criação do diretório para salvar os modelos
        os.makedirs("data/06_models", exist_ok=True)

        # Salva os modelos localmente
        logistic_path, _ = exp.save_model(tuned_logistic, "data/06_models/logistic_model")
        best_path, _ = exp.save_model(tuned_best, "data/06_models/best_classifier")

        # Faz o log dos arquivos salvos
        mlflow.log_artifact("data/06_models/logistic_model.pkl", artifact_path="modelos")
        mlflow.log_artifact("data/06_models/best_classifier.pkl", artifact_path="modelos")

        # Log de parâmetros e métricas simples
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_metric("train_rows", len(data))

    return {
        "best_model": tuned_best,
        "logistic_model": tuned_logistic
    }
