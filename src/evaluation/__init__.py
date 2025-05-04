# src/evaluation/__init__.py

"""
Evaluation package for MTEB-based MSMARCO retrieval benchmark.
Exports model wrappers and evaluation entrypoints.
"""

# Expose the wrappers and evaluation functions at package level
from .model_wrapper import SentenceTransformerWrapper, TransformerCLSWrapper
from .msmarco_eval  import evaluate_all

__all__ = [
    "SentenceTransformerWrapper",
    "TransformerCLSWrapper",
    "evaluate_all",
]
