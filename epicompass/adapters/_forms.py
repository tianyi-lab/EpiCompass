"""Structural form of a law: the same form with other constants maps to the same family (used as the hypothesis key
on DiscoverPhysics, where the benchmark fits each law's declared constants)."""
from __future__ import annotations

import ast
import hashlib
from typing import Any, Dict, List, Tuple


def _is_structural_float(v: float) -> bool:
    return float(v).is_integer() or abs(v) <= 1e-9


def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_number(node.operand)
    return False


class _Skeleton(ast.NodeTransformer):
    """Free float literals and free-constant names become `c`; integer-valued
    floats become ints; argument and local names are renumbered."""

    def __init__(self, const_names: set, helper_alias: Dict[str, str]):
        self.const_names, self.helper_alias, self.local = const_names, helper_alias, {}

    def _loc(self, name: str) -> str:
        return self.local.setdefault(name, f"v{len(self.local)}")

    def visit_FunctionDef(self, node):  # noqa: N802
        for a in list(node.args.args) + list(node.args.kwonlyargs):
            a.arg = "c" if a.arg in self.const_names else self._loc(a.arg)
        if node.args.kwarg:
            node.args.kwarg.arg = "kw"
        node.args.defaults = []
        node.args.kw_defaults = [None for _ in node.args.kw_defaults]
        node.name = self.helper_alias.get(node.name, node.name)
        node.body = [b for b in node.body if not (isinstance(b, ast.Expr)
                                                  and isinstance(getattr(b, "value", None), ast.Constant)
                                                  and isinstance(b.value.value, str))] or [ast.Pass()]
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):  # noqa: N802
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self.const_names):
            return None
        node.value = self.visit(node.value)
        for t in node.targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and n.id not in self.const_names:
                    n.id = self._loc(n.id)
        return node

    def visit_Name(self, node):  # noqa: N802
        if node.id in self.const_names:
            return ast.Name(id="c", ctx=node.ctx)
        if node.id in self.helper_alias:
            return ast.Name(id=self.helper_alias[node.id], ctx=node.ctx)
        if node.id in self.local:
            return ast.Name(id=self.local[node.id], ctx=node.ctx)
        return node

    def visit_Constant(self, node):  # noqa: N802
        if isinstance(node.value, float) and not isinstance(node.value, bool):
            if float(node.value).is_integer() and abs(node.value) < 1e15:
                return ast.Constant(value=int(node.value))
            if abs(node.value) <= 1e-9:
                return ast.Constant(value=0)
            return ast.Name(id="c", ctx=ast.Load())
        return node


def family_signature(src: str, entry: str = "predict", ignore: Tuple[str, ...] = ("fit", "fit_parameters")) -> Dict[str, Any]:
    """{"family", "skeleton", "n_free", "n_ops"} of a candidate's source.

    `entry` is the function that IS the law (`predict`, `discovered_law` ...).
    Helpers it calls are followed; functions in `ignore` (how constants are
    fitted) are not part of the form. Unparseable text hashes as raw text.
    """
    def raw(tag):
        return {"family": hashlib.sha1(src.encode()).hexdigest()[:8], "skeleton": tag, "n_free": 0, "n_ops": 0}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return raw("<unparseable>")
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry), None)
    if fn is None:
        return raw(f"<no {entry}>")
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != entry and n.name not in ignore}
    order: List[str] = []
    stack = [fn]
    while stack:
        cur = stack.pop(0)
        for n in ast.walk(cur):
            if isinstance(n, ast.Name) and n.id in defs and n.id not in order:
                order.append(n.id)
                stack.append(defs[n.id])
    helpers = [defs[n] for n in order]
    alias = {h.name: f"h{i}" for i, h in enumerate(helpers)}

    def free_consts(f):
        names = set()
        for stmt in ast.walk(f):
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and _is_number(stmt.value)):
                v = stmt.value.operand.value if isinstance(stmt.value, ast.UnaryOp) else stmt.value.value
                if isinstance(v, float) and not _is_structural_float(v):
                    names.add(stmt.targets[0].id)
        return names

    n_free = n_ops = 0
    parts = []
    for f in [fn] + helpers:
        consts = free_consts(f)
        n_free += len(consts) + sum(1 for n in ast.walk(f) if isinstance(n, ast.Constant) and isinstance(n.value, float)
                                    and not isinstance(n.value, bool) and not _is_structural_float(n.value)) - len(consts)
        n_ops += sum(1 for n in ast.walk(f) if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare)))
        node = _Skeleton(consts, alias).visit(ast.parse(ast.unparse(f)).body[0])
        ast.fix_missing_locations(node)
        parts.append("".join(ast.unparse(node).split()))
    skel = "||".join(parts)
    return {"family": hashlib.sha1(skel.encode()).hexdigest()[:8], "skeleton": skel[:200],
            "n_free": n_free, "n_ops": n_ops}
