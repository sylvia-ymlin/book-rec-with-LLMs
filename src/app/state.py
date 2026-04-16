"""
Shared app state: singleton service instances initialized at startup.

Route modules import from here instead of from main.py to avoid circular imports.
"""
from typing import Optional

# Set by startup_event() in main.py
recommender: Optional[object] = None
rec_service: Optional[object] = None
