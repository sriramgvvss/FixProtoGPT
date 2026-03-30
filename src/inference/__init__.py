"""Package: src.inference
========================

Inference engine, message enrichment, and explanation builders.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

from .generate import FixProtoGPTInference
from .enrichment import enrich_fix_message
from .explainer import build_field_explanation, build_explain_summary

__all__ = [
    "FixProtoGPTInference",
    "enrich_fix_message",
    "build_field_explanation",
    "build_explain_summary",
]
