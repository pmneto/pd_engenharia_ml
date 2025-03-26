from kedro.pipeline import Pipeline, node
from .nodes import download_and_load_data, preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(download_and_load_data, inputs=None, outputs="kobe_shots", name="download_data"),
        #node(preprocess_data, inputs="kobe_shots", outputs="preprocessed_shots", name="preprocess_data"),
    ])
