"""The model side of the agent.

The agent only needs one method, `next_step(task, history)`, that returns a
`ToolCall` or a `Final`. Plug in a real model by implementing that method.
`ScriptedLLM` replays a fixed list of steps, which makes runs repeatable and
lets tests play the part of a model that has been tricked into misbehaving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Union


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Final:
    text: str


@dataclass(frozen=True)
class Observation:
    call: ToolCall
    allowed: bool
    text: str


Step = Union[ToolCall, Final]


class LLM(Protocol):
    def next_step(self, task: str, history: List[Observation]) -> Step: ...


class ScriptedLLM:
    """Replays steps in order, then finishes."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)
        self._index = 0

    def next_step(self, task: str, history: List[Observation]) -> Step:
        if self._index >= len(self._steps):
            return Final("done")
        step = self._steps[self._index]
        self._index += 1
        return step
