from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_data_filtered,
    train_logistic_model,
    train_best_classifier,
    train_decision_tree_model
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Visualização inicial dos dados brutos com shot_made_flag
        node(
            func=plot_data_filtered,
            inputs="filtered_shots",
            outputs=None,
            name="plot_raw_data_before_training"
        ),

        # Treinamento com Regressão Logística
        node(
            func=train_logistic_model,
            inputs={
                "data": "filtered_shots",
                "session_id": "params:session_id",
                "cv_folds": "params:cv_folds"
            },
            outputs="logistic_model",
            name="train_logistic_model_node"
        ),

        # Treinamento com o melhor classificador automático (excluindo regressão logística)
        node(
            func=train_best_classifier,
            inputs={
                "data": "filtered_shots",
                "session_id": "params:session_id",
                "cv_folds": "params:cv_folds"
            },
            outputs="best_model",
            name="train_best_classifier_node"
        ),

        # Treinamento com árvore de decisão (exigido pelo enunciado)
        node(
            func=train_decision_tree_model,
            inputs={
                "data": "filtered_shots",
                "session_id": "params:session_id",
                "cv_folds": "params:cv_folds"
            },
            outputs="decisionTree",
            name="train_decision_tree_model_node"
        )
    ])
