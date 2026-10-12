"""Minimal control-flow-graph builder over the supported program subset."""
from __future__ import annotations

import ast


class _CFG:
    def __init__(self) -> None:
        self.stmts: list[int] = []
        self.edges: list[tuple[int, int]] = []
        self.back_edges = 0
        self.max_depth = 0

    def new(self) -> int:
        self.stmts.append(0)
        return len(self.stmts) - 1

    def edge(self, a: int, b: int) -> None:
        self.edges.append((a, b))

    def seq(self, stmts: list[ast.stmt], cur: int, depth: int) -> int:
        self.max_depth = max(self.max_depth, depth)
        for st in stmts:
            if isinstance(st, ast.For):
                head, body = self.new(), self.new()
                self.edge(cur, head)
                self.edge(head, body)
                end = self.seq(st.body, body, depth + 1)
                self.edge(end, head)
                self.back_edges += 1
                after = self.new()
                self.edge(head, after)
                cur = after
            elif isinstance(st, ast.If):
                self.stmts[cur] += 1  # condition evaluation
                then = self.new()
                self.edge(cur, then)
                then_end = self.seq(st.body, then, depth)
                join = self.new()
                self.edge(then_end, join)
                if st.orelse:
                    els = self.new()
                    self.edge(cur, els)
                    self.edge(self.seq(st.orelse, els, depth), join)
                else:
                    self.edge(cur, join)
                cur = join
            else:
                self.stmts[cur] += 1
        return cur


def cfg_stats(tree: ast.Module) -> dict[str, float]:
    g = _CFG()
    entry = g.new()
    g.seq(tree.body[0].body, entry, 0)
    n, e = len(g.stmts), len(g.edges)
    return {
        "cfg_blocks": float(n),
        "cfg_edges": float(e),
        "cfg_cyclomatic": float(e - n + 2),
        "cfg_back_edges": float(g.back_edges),
        "cfg_max_depth": float(g.max_depth),
        "cfg_max_block_stmts": float(max(g.stmts)),
        "cfg_mean_block_stmts": float(sum(g.stmts) / n),
    }
