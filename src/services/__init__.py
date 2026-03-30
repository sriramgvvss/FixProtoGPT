"""Package: src.services
========================

Application-level services — update management, spec monitoring,
and orchestration tasks.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

from .update_manager import UpdateManager

__all__ = [
    "UpdateManager",
]
