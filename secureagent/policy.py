"""Policy: what a single agent run is allowed to do."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, FrozenSet

# Permissions a tool can require. A run only gets the ones listed in Policy.granted.
FS_READ = "fs.read"
FS_WRITE = "fs.write"
COMPUTE = "compute"

# Called as approver(tool_name, args, reason). Return True to allow the call.
Approver = Callable[[str, dict, str], bool]


def deny_all(tool: str, args: dict, reason: str) -> bool:
    """Default approver: nobody is watching, so the answer is no."""
    return False


@dataclass(frozen=True)
class Policy:
    sandbox: Path
    granted: FrozenSet[str] = frozenset()
    max_calls: int = 10
    max_output_chars: int = 2000
    # Once the agent has read untrusted data, side-effect tools need approval.
    approve_after_untrusted: bool = True
    approver: Approver = deny_all

    def __post_init__(self) -> None:
        # Resolve once so every later containment check compares real paths.
        object.__setattr__(self, "sandbox", Path(self.sandbox).resolve())
