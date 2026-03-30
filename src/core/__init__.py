"""Package: src.core
====================

Core domain: model architecture, tokenizer, FIX protocol reference
data, and version registry.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

from .transformer import (
    FixProtoGPT,
    ModelConfig,
    create_model,
)
from .tokenizer import FixProtocolTokenizer
from .fix_reference import (
    MESSAGE_TYPES,
    FIELDS,
    ENUMERATIONS,
    DATA_TYPES,
    COMPONENTS,
)
from .version_registry import (
    VERSIONS,
    get_version_info,
    is_valid_version,
    default_version,
    list_installed,
)

__all__ = [
    "FixProtoGPT",
    "ModelConfig",
    "create_model",
    "FixProtocolTokenizer",
    "MESSAGE_TYPES",
    "FIELDS",
    "ENUMERATIONS",
    "DATA_TYPES",
    "COMPONENTS",
    "VERSIONS",
    "get_version_info",
    "is_valid_version",
    "default_version",
    "list_installed",
]
