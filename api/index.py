"""Vercel serverless fallback entrypoint (re-exports the main FastAPI app)."""

from app import app

__all__ = ["app"]