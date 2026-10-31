"""HTTP routers. Each module exposes a module-level ``router``."""

from app.api.routes import chat, documents, health, jobs

__all__ = ["chat", "documents", "health", "jobs"]
