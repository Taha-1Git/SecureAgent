"""SecureAgent: a small agent loop with a permission guard around every tool call."""
from .agent import Agent, RunResult
from .audit import AuditLog
from .guard import Decision, Guard, RunState
from .llm import Final, Observation, ScriptedLLM, ToolCall
from .policy import COMPUTE, FS_READ, FS_WRITE, Policy
from .tools import ToolRegistry, ToolSpec, default_registry

__all__ = [
    "Agent", "RunResult", "AuditLog", "Decision", "Guard", "RunState",
    "Final", "Observation", "ScriptedLLM", "ToolCall",
    "COMPUTE", "FS_READ", "FS_WRITE", "Policy",
    "ToolRegistry", "ToolSpec", "default_registry",
]
