from kedro.pipeline import Pipeline, node, pipeline
from .nodes import aplicar_modelo_prod

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=aplicar_modelo_prod,
            inputs={
                "model_path": "logistic_model",
                "df_prod": "prod_shots"
            },
            outputs="df_predito",
            name="aplicar_modelo_producao_node"
        )
    ])
