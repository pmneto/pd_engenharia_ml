from pycaret.classification import ClassificationExperiment
import mlflow
import os

def train_and_save_models(data, session_id: int):
    exp = ClassificationExperiment()

    with mlflow.start_run(run_name="TreinamentoModelos"):

        exp.setup(
            data=data,
            target="shot_made_flag",
            session_id=session_id,
            train_size=0.8,
            fold=5,
            fold_shuffle=True,
            html=False,
            log_experiment=True,
            experiment_name="kobe_classificacao",
            log_plots=False,
            verbose=False
        )

        # 1. Regressão logística
        logistic = exp.create_model("lr", fold=5)
        tuned_logistic = exp.tune_model(logistic, n_iter=10, fold=5, optimize="AUC")

        # 2. Melhor modelo classificador (exceto regressão)
        best_model = exp.compare_models(exclude=["lr"], fold=3, n_select=1)
        tuned_best = exp.tune_model(best_model, n_iter=10, fold=3, optimize="AUC")

        # Salva os modelos localmente
        os.makedirs("data/06_models", exist_ok=True)
        exp.save_model(tuned_logistic, "data/06_models/logistic_model")
        exp.save_model(tuned_best, "data/06_models/best_classifier")

        # Log extra no MLflow
        mlflow.log_param("cv_folds_logistic", 5)
        mlflow.log_param("cv_folds_best_model", 3)

        return tuned_best  # Para servir depois
