"""Append-only record of every decision the guard makes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .llm import ToolCall

MAX_VALUE_CHARS = 80


def summarize_args(args: Dict[str, object]) -> Dict[str, str]:
    """Log what was asked for without copying file contents into the log."""
    summary = {}
    for key, value in args.items():
        if key == "content" and isinstance(value, str):
            summary[key] = f"<{len(value)} chars>"
        else:
            text = repr(value)
            summary[key] = text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "..."
    return summary


class AuditLog:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.events: List[dict] = []
        self._path = Path(path) if path else None

    def record(self, call: ToolCall, allowed: bool, reason: str) -> None:
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "tool": call.name,
            "args": summarize_args(call.args),
            "allowed": allowed,
            "reason": reason,
        }
        self.events.append(event)
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
