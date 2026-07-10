"""A-share factor analysis project package."""

from factor_analysis.config import PipelineConfig

__all__ = ["PipelineConfig", "run_pipeline"]


def run_pipeline(config: PipelineConfig):
    from factor_analysis.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(config)
