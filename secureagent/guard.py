"""The guard: decides whether a tool call may run.

It never asks the model for permission or trusts anything the model says about
itself. Every check uses only the call, the tool's declaration and the policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import ToolCall
from .policy import Policy
from .tools import ToolRegistry


@dataclass
class RunState:
    calls: int = 0          # tool calls attempted so far, allowed or not
    tainted: bool = False   # has the agent read data an attacker could control?


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def _deny(reason: str) -> Decision:
    return Decision(False, reason)


class Guard:
    def __init__(self, registry: ToolRegistry, policy: Policy) -> None:
        self.registry = registry
        self.policy = policy

    def check(self, call: ToolCall, state: RunState) -> Decision:
        spec = self.registry.get(call.name)
        if spec is None:
            return _deny("unknown tool")

        if state.calls > self.policy.max_calls:
            return _deny("call budget exhausted")

        missing = spec.permissions - self.policy.granted
        if missing:
            return _deny("missing permission: " + ", ".join(sorted(missing)))

        problem = self._argument_problem(spec.params, call.args)
        if problem:
            return _deny(problem)

        if spec.side_effect and state.tainted and self.policy.approve_after_untrusted:
            reason = "side-effect tool after reading untrusted data"
            if not self.policy.approver(call.name, dict(call.args), reason):
                return _deny("approval required: " + reason)
            return Decision(True, "approved after untrusted read")

        return Decision(True, "allowed by policy")

    @staticmethod
    def _argument_problem(params: dict, args: dict) -> str:
        if not isinstance(args, dict):
            return "arguments must be a dict"
        unexpected = set(args) - set(params)
        if unexpected:
            return "unexpected argument: " + ", ".join(sorted(map(str, unexpected)))
        for name, expected in params.items():
            if name not in args:
                return f"missing argument: {name}"
            if not isinstance(args[name], expected):
                return f"argument {name} must be {expected.__name__}"
        return ""
