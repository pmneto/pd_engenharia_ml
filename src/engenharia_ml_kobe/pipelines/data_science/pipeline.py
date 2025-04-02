from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_and_save_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_and_save_models,
            inputs={
                "data": "filtered_shots",
                "session_id": "params:session_id",
                "cv_folds": "params:cv_folds"
            },
            outputs={
                "best_model": "best_model",
                "logistic_model": "logistic_model"
            },
            name="train_models_node"
        )
    ])
