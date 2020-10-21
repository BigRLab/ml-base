"""Base classes and utilities that are useful for deploying ML models."""

from ml_base.ml_model import MLModel
from ml_base.decorator import MLModelDecorator

__version_info__ = ("0", "2", "0")
__version__ = ".".join(__version_info__)

__all__ = ["MLModel", "MLModelDecorator"]
