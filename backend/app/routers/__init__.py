"""Aggregate and expose router modules for FastAPI.

This file ensures `app.routers` is recognised as a package and allows
`from app.routers import <router>` imports in `app.main`.
"""

__all__ = [
    "health",
    "analytics",
    "maneuvers",
    "satellites",
    "advanced",
    "dashboard",
    "ccdm",
    "trajectory",
    "comparison",
    "events",
    "diagnostics",
] 