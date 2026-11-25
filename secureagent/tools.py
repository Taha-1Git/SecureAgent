"""Tool definitions and the built-in tools."""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional

from .policy import COMPUTE, FS_READ, FS_WRITE

MAX_FILE_BYTES = 100_000
MAX_LIST_ENTRIES = 200
MAX_EXPRESSION_CHARS = 200


class ToolError(Exception):
    """A tool failed for an ordinary reason (missing file, bad expression)."""


class Blocked(Exception):
    """A tool refused an argument because it crosses a security boundary."""


@dataclass(frozen=True)
class ToolContext:
    sandbox: Path


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permissions: FrozenSet[str]      # all of these must be granted by the policy
    params: Dict[str, type]          # exact argument names and types
    func: Callable[..., str]         # func(ctx, **args) -> str
    side_effect: bool = False        # changes the outside world
    untrusted_output: bool = False   # output may carry attacker-controlled text


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)


def safe_path(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`, refusing anything that lands outside it.

    Resolving first means `..`, absolute paths and symlinks are all judged by
    where they actually point, not by how they are written.
    """
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise Blocked("invalid path")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise Blocked("path escapes the sandbox")
    return candidate


# --- built-in tools -------------------------------------------------------

def _read_file(ctx: ToolContext, path: str) -> str:
    target = safe_path(ctx.sandbox, path)
    if not target.is_file():
        raise ToolError("not a file")
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read(MAX_FILE_BYTES)


def _list_dir(ctx: ToolContext, path: str) -> str:
    target = safe_path(ctx.sandbox, path)
    if not target.is_dir():
        raise ToolError("not a directory")
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(names[:MAX_LIST_ENTRIES])


def _write_file(ctx: ToolContext, path: str, content: str) -> str:
    if len(content) > MAX_FILE_BYTES:
        raise ToolError("content too large")
    target = safe_path(ctx.sandbox, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if not (0 <= right <= 10 and abs(left) <= 1_000_000):
                raise ToolError("exponent out of range")
            return left ** right
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError("operator not allowed")
        return op(left, right)
    raise ToolError("expression not allowed")


def _calculator(ctx: ToolContext, expression: str) -> str:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree))
    except SyntaxError:
        raise ToolError("could not parse expression")
    except ZeroDivisionError:
        raise ToolError("division by zero")


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "read_file", "Read a text file inside the sandbox.",
        frozenset({FS_READ}), {"path": str}, _read_file, untrusted_output=True))
    registry.register(ToolSpec(
        "list_dir", "List a directory inside the sandbox.",
        frozenset({FS_READ}), {"path": str}, _list_dir, untrusted_output=True))
    registry.register(ToolSpec(
        "write_file", "Write a text file inside the sandbox.",
        frozenset({FS_WRITE}), {"path": str, "content": str}, _write_file,
        side_effect=True))
    registry.register(ToolSpec(
        "calculator", "Evaluate a plain arithmetic expression.",
        frozenset({COMPUTE}), {"expression": str}, _calculator))
    return registry
