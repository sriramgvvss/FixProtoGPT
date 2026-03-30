"""Package: src.persistence
===========================

SQLite-backed persistence layer for user management and
interaction logging.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

from .interaction_logger import InteractionLogger
from .user_manager import UserManager

__all__ = [
    "InteractionLogger",
    "UserManager",
]
