"""
Langfuse setup and session ID management (langfuse v4).
Session ID is set via LANGFUSE_SESSION_ID env var before importing langfuse.
"""
from __future__ import annotations
import os
import uuid
from dotenv import load_dotenv

load_dotenv()


def new_session_id() -> str:
    """Generate a unique session ID for this pipeline run."""
    return f"reply-mirror-{uuid.uuid4().hex[:16]}"


def configure(session_id: str) -> None:
    """
    Set the Langfuse session ID in env before any langfuse imports take effect.
    Must be called before importing any module that uses @observe().
    """
    os.environ["LANGFUSE_SESSION_ID"] = session_id
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY", ""))
    os.environ.setdefault(
        "LANGFUSE_HOST",
        os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )
