"""Package: src
==============

FixProtoGPT — FIX Protocol Language Model.

Coding Standards: PEP 8, PEP 257, PEP 484, Google Python Style Guide.
"""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("fixprotogpt")
except Exception:
    __version__ = "1.0.0"  # fallback when not installed as a package
