"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from engenharia_ml_kobe.pipelines.data_engineering.pipeline import create_pipeline as de_pipeline
from engenharia_ml_kobe.pipelines.data_science.pipeline import create_pipeline as ds_pipeline
from engenharia_ml_kobe.pipelines.application.pipeline import create_pipeline as app_pipeline


#from kobe_shots.pipelines.model_training.pipeline import create_pipeline as mt_pipeline
#from kobe_shots.pipelines.model_evaluation.pipeline import create_pipeline as me_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "de": de_pipeline(),  # Pipeline de Data Engineering
        "ds": ds_pipeline(),  # Pipeline de Data Science
        "app":app_pipeline(), # Pipeline de Application
        "__default__": de_pipeline() + ds_pipeline() + app_pipeline()
    }