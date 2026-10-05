"""Private local dashboard adapter. Does not start collectors or Telegram workers."""

from app.web.server import create_app

__all__ = ["create_app"]
