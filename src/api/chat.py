"""
Backward-compatible shim for chat API.

The actual implementation now lives in `src.app.api.chat`.
Import `router` from there so existing imports of `src.api.chat` keep working.
"""

from src.app.api.chat import router  # noqa: F401

__all__ = ["router"]
