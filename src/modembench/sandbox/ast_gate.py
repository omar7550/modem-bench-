"""Static AST policy for receiver source. Defense in depth; the kernel profile is the boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Iterator

MAX_SOURCE_BYTES = 256 * 1024
ALLOW_STDLIB_MATH = True
ALLOW_DUNDER_DEFINITIONS = True
AST_POLICY_VERSION = "modembench-ast-v5-math1-dunderdef1-noframes-libnarrowed"

_DUNDER = re.compile(r"^__.*__$")
MAIN_GUARD_NAME = "__name__"
# scope-opening nodes: __name__ is only exempt at module level
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
# `lib` is not banned: np.lib.stride_tricks is legitimate compute; the IO in it is
# banned per-function below
_BANNED_NUMPY_SUBMODULES = {
    "core",
    "ctypeslib",
    "distutils",
    "f2py",
    "npyio",
    "testing",
}
_BANNED_NUMPY_MEMBERS = {
    "DataSource",
    "ctypeslib",
    "f2py",
    "fromfile",
    "fromregex",
    "genfromtxt",
    "load",
    "loadtxt",
    "memmap",
    # np.lib.format IO named per-function; banning the attribute `format` would
    # also reject str.format
    "npyio",
    "open_memmap",
    "read_array",
    "read_magic",
    "write_array",
    "read_magic",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "testing",
}
# Frame/code/traceback introspection: no underscore, no dunder, so no other rule sees
# these, yet gi_frame.f_builtins['__import__'] escapes.
_FRAME_INTROSPECTION_ATTRIBUTES = {
    "ag_await", "ag_code", "ag_frame",
    "cr_await", "cr_code", "cr_frame", "cr_origin",
    "gi_code", "gi_frame", "gi_yieldfrom",
    "f_back", "f_builtins", "f_code", "f_globals", "f_lasti", "f_lineno",
    "f_locals", "f_trace",
    "co_code", "co_consts", "co_filename", "co_freevars", "co_names",
    "co_varnames",
    "tb_frame", "tb_next",
    "func_code", "func_globals",
    "cell_contents",
}
_BANNED_ATTRIBUTES = _BANNED_NUMPY_MEMBERS | _FRAME_INTROSPECTION_ATTRIBUTES | {
    "distutils",
    "dump",
    "dumps",
    "tofile",
}
_BANNED_BUILTINS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}


@dataclass(frozen=True)
class Violation:
    lineno: int
    rule: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"lineno": self.lineno, "rule": self.rule, "detail": self.detail}


def _module_level_compares(node: ast.AST) -> Iterator[ast.Compare]:
    """Every ``Compare`` reachable from ``node`` without crossing into a nested scope."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPE_NODES):
            continue
        if isinstance(child, ast.Compare):
            yield child
        yield from _module_level_compares(child)


@dataclass(frozen=True)
class _DunderExemptions:
    """Exempt dunder binding sites, keyed by node identity so a name elsewhere reopens nothing.

    Only two forms: a def directly in a ClassDef body (name only; args/body still checked),
    and __name__ as a direct operand of a module-level Compare. Dunder reads stay banned.
    """

    method_defs: frozenset[ast.AST]
    main_guard_names: frozenset[ast.AST]

    @classmethod
    def none(cls) -> "_DunderExemptions":
        return cls(frozenset(), frozenset())

    @classmethod
    def locate(cls, tree: ast.AST) -> "_DunderExemptions":
        method_defs = {
            statement
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        main_guard_names = {
            operand
            for compare in _module_level_compares(tree)
            for operand in (compare.left, *compare.comparators)
            if isinstance(operand, ast.Name)
            and operand.id == MAIN_GUARD_NAME
            and isinstance(operand.ctx, ast.Load)
        }
        return cls(frozenset(method_defs), frozenset(main_guard_names))


def _module_violation(module: str, *, allow_stdlib_math: bool) -> tuple[str, str] | None:
    components = module.split(".")
    if any(component.startswith("_") for component in components):
        return "private_module_component", f"private module component in {module!r}"
    if module == "numpy" or module.startswith("numpy."):
        if len(components) > 1 and components[1] in _BANNED_NUMPY_SUBMODULES:
            return "numpy_submodule_banned", f"numpy submodule {module!r} is banned"
        return None
    if module == "scipy.signal" or module.startswith("scipy.signal."):
        return None
    if allow_stdlib_math and module in {"math", "cmath"}:
        return None
    return "import_not_allowed", f"module {module!r} is not allowed"


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, allow_stdlib_math: bool, exemptions: _DunderExemptions) -> None:
        self.allow_stdlib_math = allow_stdlib_math
        self.exemptions = exemptions
        self.violations: list[Violation] = []

    def add(self, node: ast.AST, rule: str, detail: str) -> None:
        self.violations.append(Violation(getattr(node, "lineno", 0), rule, detail))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            violation = _module_violation(alias.name, allow_stdlib_math=self.allow_stdlib_math)
            if violation is not None:
                self.add(node, *violation)
            if alias.asname and _DUNDER.match(alias.asname):
                self.add(node, "dunder_identifier", f"dunder import alias {alias.asname!r} is banned")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            self.add(node, "import_not_allowed", "relative imports are not allowed")
        elif module == "scipy":
            for alias in node.names:
                if alias.name != "signal":
                    self.add(node, "import_not_allowed", f"scipy member {alias.name!r} is not allowed")
        else:
            violation = _module_violation(module, allow_stdlib_math=self.allow_stdlib_math)
            if violation is not None:
                self.add(node, *violation)
        for alias in node.names:
            if alias.name == "*":
                self.add(node, "wildcard_import", "wildcard imports are not allowed")
            if module == "numpy" and alias.name in _BANNED_NUMPY_MEMBERS:
                self.add(node, "import_member_banned", f"numpy member {alias.name!r} is banned")
            if alias.name.startswith("_"):
                self.add(node, "private_module_component", f"private import member {alias.name!r}")
            if alias.asname and _DUNDER.match(alias.asname):
                self.add(node, "dunder_identifier", f"dunder import alias {alias.asname!r} is banned")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _DUNDER.match(node.name) and node not in self.exemptions.method_defs:
            self.add(
                node,
                "dunder_identifier",
                f"dunder function name {node.name!r} is banned outside a class body",
            )
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if _DUNDER.match(node.name):
            self.add(node, "dunder_identifier", f"dunder class name {node.name!r} is banned")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if _DUNDER.match(node.arg):
            self.add(node, "dunder_identifier", f"dunder argument {node.arg!r} is banned")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # never exempt: dunder attribute reads are the introspection escape
        if _DUNDER.match(node.attr):
            self.add(node, "dunder_identifier", f"dunder attribute {node.attr!r} is banned")
        elif node.attr.startswith("_"):
            self.add(node, "private_attribute", f"private attribute {node.attr!r} is banned")
        elif node.attr in _FRAME_INTROSPECTION_ATTRIBUTES:
            self.add(node, "frame_introspection", f"frame/code attribute {node.attr!r} is banned")
        elif node.attr in _BANNED_ATTRIBUTES:
            self.add(node, "numpy_io_attribute", f"attribute {node.attr!r} is banned")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if _DUNDER.match(node.id) and node not in self.exemptions.main_guard_names:
            self.add(node, "dunder_identifier", f"dunder name {node.id!r} is banned")
        elif node.id in _BANNED_BUILTINS:
            self.add(node, "builtin_banned", f"builtin {node.id!r} is banned")
        self.generic_visit(node)


def check_source(
    source: bytes | str,
    *,
    allow_stdlib_math: bool | None = None,
    allow_dunder_definitions: bool | None = None,
) -> dict[str, Any]:
    """Return the structured AST verdict for raw receiver source."""
    raw = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    digest = sha256(raw).hexdigest()
    if len(raw) > MAX_SOURCE_BYTES:
        violation = Violation(0, "source_size", f"source exceeds {MAX_SOURCE_BYTES} bytes")
        return {"ok": False, "violations": [violation.as_dict()], "source_sha256": digest}
    try:
        text = raw.decode("utf-8")
        tree = ast.parse(text, filename="receiver.py")
    except (UnicodeDecodeError, SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        lineno = getattr(exc, "lineno", 0) or 0
        violation = Violation(lineno, "parse_error", f"{type(exc).__name__}: {exc}")
        return {"ok": False, "violations": [violation.as_dict()], "source_sha256": digest}
    allowed_math = ALLOW_STDLIB_MATH if allow_stdlib_math is None else allow_stdlib_math
    allowed_definitions = (
        ALLOW_DUNDER_DEFINITIONS
        if allow_dunder_definitions is None
        else allow_dunder_definitions
    )
    exemptions = (
        _DunderExemptions.locate(tree) if allowed_definitions else _DunderExemptions.none()
    )
    visitor = _PolicyVisitor(allowed_math, exemptions)
    visitor.visit(tree)
    violations = sorted(visitor.violations, key=lambda item: (item.lineno, item.rule, item.detail))
    return {
        "ok": not violations,
        "violations": [item.as_dict() for item in violations],
        "source_sha256": digest,
    }
