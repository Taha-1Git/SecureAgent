"""The agent loop: ask the model for a step, check it, run it, report back."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .audit import AuditLog
from .guard import Guard, RunState
from .llm import LLM, Final, Observation, ToolCall
from .policy import Policy
from .tools import Blocked, ToolContext, ToolError, ToolRegistry


@dataclass
class RunResult:
    final: Optional[str]
    observations: List[Observation]
    stopped: str            # "final" or "budget"


class Agent:
    def __init__(self, llm: LLM, registry: ToolRegistry, policy: Policy,
                 audit: Optional[AuditLog] = None) -> None:
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.audit = audit if audit is not None else AuditLog()
        self.guard = Guard(registry, policy)

    def run(self, task: str) -> RunResult:
        state = RunState()
        context = ToolContext(self.policy.sandbox)
        history: List[Observation] = []

        while True:
            if state.calls >= self.policy.max_calls:
                return RunResult(None, history, "budget")

            step = self.llm.next_step(task, history)
            if isinstance(step, Final):
                return RunResult(step.text, history, "final")
            if not isinstance(step, ToolCall):
                raise TypeError("model must return a ToolCall or a Final")

            state.calls += 1
            history.append(self._attempt(step, state, context))

    def _attempt(self, call: ToolCall, state: RunState, context: ToolContext) -> Observation:
        decision = self.guard.check(call, state)
        if not decision.allowed:
            self.audit.record(call, False, decision.reason)
            return Observation(call, False, "DENIED: " + decision.reason)

        spec = self.registry.get(call.name)
        try:
            output = spec.func(context, **call.args)
        except Blocked as exc:
            self.audit.record(call, False, f"blocked by tool: {exc}")
            return Observation(call, False, f"DENIED: {exc}")
        except ToolError as exc:
            self.audit.record(call, True, f"{decision.reason}; tool error: {exc}")
            return Observation(call, True, f"ERROR: {exc}")
        except Exception as exc:  # a crashing tool must not crash the agent
            self.audit.record(call, True, f"{decision.reason}; tool crashed: {type(exc).__name__}")
            return Observation(call, True, "ERROR: tool failed")

        self.audit.record(call, True, decision.reason)
        if spec.untrusted_output:
            state.tainted = True

        limit = self.policy.max_output_chars
        if len(output) > limit:
            output = output[:limit] + "\n[truncated]"
        return Observation(call, True, output)
