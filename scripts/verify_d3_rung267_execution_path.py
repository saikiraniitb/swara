#!/usr/bin/env python3
"""Static fail-closed guard against routing D3-267 to D2 full batching."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_swara_d3_data_scaling.py"
MICROBATCH = ROOT / "scripts/run_swara_d3_microbatch.py"


def is_rung267_test(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Attribute)
        and isinstance(test.left.value, ast.Name)
        and test.left.value.id == "args"
        and test.left.attr == "rung"
        and len(test.ops) == len(test.comparators) == 1
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == 267
    )


def main() -> None:
    wrapper_source = WRAPPER.read_text(encoding="utf-8")
    micro_source = MICROBATCH.read_text(encoding="utf-8")
    tree = ast.parse(wrapper_source, filename=str(WRAPPER))
    route = next((node for node in ast.walk(tree) if isinstance(node, ast.If) and is_rung267_test(node)), None)
    if route is None:
        raise RuntimeError("D3-267 dispatch guard missing")
    route_source = ast.get_source_segment(wrapper_source, route) or ""
    if "run_rung267" not in route_source or "return" not in route_source:
        raise RuntimeError("D3-267 dispatch does not terminate in the microbatch runner")
    if "d2.main" in route_source:
        raise RuntimeError("D3-267 dispatch reaches legacy D2 full-batch main")
    if "d2.main" in micro_source:
        raise RuntimeError("D3 microbatch runner reaches legacy D2 full-batch main")
    required = (
        "logical_microbatch_step",
        "for microbatch in chunks(examples, microbatch_rows)",
        "contribution.backward()",
        "optimizer.step()",
        "maximum_autograd_rows",
    )
    missing = [item for item in required if item not in micro_source]
    if missing:
        raise RuntimeError(f"D3 microbatch structural contract missing: {missing}")
    print(json.dumps({
        "D3_RUNG267_EXECUTION_PATH": "PASS",
        "wrapper": "run_swara_d3_data_scaling.py --rung 267 -> run_rung267",
        "training_path": "run_swara_d3_microbatch.run_rung267/logical_microbatch_step",
        "legacy_d2_main_reachable_for_rung267": False,
    }, indent=2))


if __name__ == "__main__":
    main()
