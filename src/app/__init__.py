"""
Application package for FastAPI entrypoint and API routers.

This module exists to make `src.app` a proper package so we can mount
`main.py` under `src/app/main.py` while keeping `src.main:app` as the
public entrypoint via a thin shim.
"""

