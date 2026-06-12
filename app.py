"""Root re-export for local tools; Vercel uses api/index.py via pyproject.toml."""

from api.index import app

__all__ = ["app"]