from kedro.pipeline import Pipeline, node
from .nodes import download_and_load_data, preprocess_data,download_and_load_data_prod

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(download_and_load_data, inputs=None, outputs="kobe_shots", name="download_data"),
        node(download_and_load_data_prod, inputs=None, outputs="prod_shots", name="download_data_prod"),
        node(preprocess_data, inputs="kobe_shots", outputs="preprocessed_shots", name="preprocess_data"),
    ])
