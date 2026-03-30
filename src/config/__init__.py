"""
Package: src.config
====================

Environment-aware configuration for FixProtoGPT.

Usage::

    from src.config.env_config import env

    env.SECRET_KEY      # the Flask secret key
    env.DEBUG           # True / False
    env.ENV_NAME        # "dev", "qa", "preprod", "prod"
    env.SEED_DEMO_USERS # whether to create default demo accounts
"""
