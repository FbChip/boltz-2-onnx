"""Compiler-oriented analysis of an fp16 ONNX graph.

Usage:
    python tools/analyze_diffusion_graph.py
    python tools/analyze_diffusion_graph.py path/to/diffusion_step_fp16.onnx
    python tools/analyze_diffusion_graph.py path/to/trunk_fp16.onnx

The graph is loaded without external tensor data because this analysis only
needs graph structure, initializers, and value metadata.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import onnx


DEFAULT_GRAPH = Path(__file__).resolve().parents[1] / "fp16" / "diffusion_step_fp16.onnx"
ACCELERATOR_OPS = {"MatMul", "LayerNormalization", "Softmax"}


def shape_of(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    result: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            result.append(dim.dim_value)
        elif dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append("?")
    return tuple(result)


def all_shapes(model: onnx.ModelProto) -> dict[str, tuple[int | str, ...]]:
    values: dict[str, tuple[int | str, ...]] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        values[value.name] = shape_of(value)
    for initializer in model.graph.initializer:
        values.setdefault(initializer.name, tuple(initializer.dims))
    return values


def numeric_product(shape: Iterable[int | str]) -> int | None:
    product = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        product *= dim
    return product


def dtype_bytes(model: onnx.ModelProto, name: str) -> int:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return {
                onnx.TensorProto.FLOAT: 4,
                onnx.TensorProto.FLOAT16: 2,
                onnx.TensorProto.DOUBLE: 8,
                onnx.TensorProto.INT64: 8,
                onnx.TensorProto.INT32: 4,
                onnx.TensorProto.INT16: 2,
                onnx.TensorProto.INT8: 1,
                onnx.TensorProto.UINT8: 1,
                onnx.TensorProto.BOOL: 1,
            }.get(initializer.data_type, 2)
    return 2


def shape_text(shape: tuple[int | str, ...] | None) -> str:
    return str(list(shape)) if shape is not None else "[unknown]"


def matmul_flops(a: tuple[int | str, ...] | None, b: tuple[int | str, ...] | None) -> int | None:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    m, k_a, k_b, n = a[-2], a[-1], b[-2], b[-1]
    if not all(isinstance(x, int) for x in (m, k_a, k_b, n)) or k_a != k_b:
        return None
    batch_a = numeric_product(a[:-2]) or 1
    batch_b = numeric_product(b[:-2]) or 1
    return 2 * batch_a * batch_b * m * n * k_a


def einsum_kind(equation: str) -> str:
    normalized = equation.replace(" ", "")
    lhs, _, rhs = normalized.partition("->")
    terms = lhs.split(",")
    if normalized in {"bihd,bjhd->bhij", "bhij,bjhd->bihd"}:
        return "Attention"
    if len(terms) == 2 and len(set(terms[0]) & set(terms[1])) == 1 and len(rhs) == 2:
        return "MatMul"
    if len(terms) == 2 and "..." in normalized and "->" in normalized:
        return "Batched MatMul"
    if "..." in normalized and len(terms) >= 2:
        return "Attention" if any(token in normalized for token in ("q", "k", "v")) else "Tensor contraction"
    return "Tensor contraction"


def node_inputs_outputs(node: onnx.NodeProto, shapes: dict[str, tuple[int | str, ...]]) -> tuple[str, str]:
    inputs = ", ".join(f"{name}{shape_text(shapes.get(name))}" for name in node.input if name)
    outputs = ", ".join(f"{name}{shape_text(shapes.get(name))}" for name in node.output if name)
    return inputs, outputs


def report_matmuls(model: onnx.ModelProto, shapes: dict[str, tuple[int | str, ...]]) -> None:
    rows = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul":
            continue
        a = shapes.get(node.input[0])
        b = shapes.get(node.input[1])
        output = shapes.get(node.output[0])
        rows.append((matmul_flops(a, b) or -1, index, a, b, output))
    print("\n## MatMul inventory (sorted by estimated FLOPs)")
    for flops, index, a, b, output in sorted(rows, reverse=True):
        estimate = f"{flops:,}" if flops >= 0 else "unknown"
        print(f"{index:5d}  FLOPs={estimate:>16}  A={shape_text(a)}  B={shape_text(b)}  Y={shape_text(output)}")
    print(f"Total MatMul nodes: {len(rows)}")


def report_einsums(model: onnx.ModelProto, shapes: dict[str, tuple[int | str, ...]]) -> None:
    print("\n## Einsum inventory")
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        equation = next((attribute.s.decode() for attribute in node.attribute if attribute.name == "equation"), "?")
        inputs, outputs = node_inputs_outputs(node, shapes)
        print(f"{index:5d}  equation={equation!r}  lowers_to={einsum_kind(equation)}")
        print(f"       inputs:  {inputs}")
        print(f"       output:  {outputs}")


def element_count(shape: tuple[int | str, ...] | None) -> int | None:
    return numeric_product(shape) if shape else None


def estimate_node_flops(node: onnx.NodeProto, shapes: dict[str, tuple[int | str, ...]]) -> int | None:
    if node.op_type in {"MatMul", "Gemm"} and len(node.input) >= 2:
        return matmul_flops(shapes.get(node.input[0]), shapes.get(node.input[1]))
    if node.op_type == "Einsum":
        equation = next((attribute.s.decode() for attribute in node.attribute if attribute.name == "equation"), "")
        output = element_count(shapes.get(node.output[0]))
        if output is not None and equation:
            return output * max(1, len(node.input))
    output = element_count(shapes.get(node.output[0])) if node.output else None
    if output is None:
        return None
    if node.op_type in {"Add", "Mul", "Sub", "Div", "Sigmoid", "Relu", "Sqrt", "Exp", "Log", "Clip", "Cast"}:
        return output
    if node.op_type in {"Softmax", "LayerNormalization", "ReduceSum", "ReduceMax", "Pow"}:
        return output * 5
    return None


def report_costs(model: onnx.ModelProto, shapes: dict[str, tuple[int | str, ...]]) -> None:
    costs: dict[str, list[int | None]] = defaultdict(list)
    traffic: dict[str, list[int | None]] = defaultdict(list)
    for node in model.graph.node:
        costs[node.op_type].append(estimate_node_flops(node, shapes))
        bytes_moved = 0
        known = True
        for name in list(node.input) + list(node.output):
            count = element_count(shapes.get(name))
            if count is None:
                known = False
                break
            bytes_moved += count * dtype_bytes(model, name)
        traffic[node.op_type].append(bytes_moved if known else None)
    total_flops = sum(value for values in costs.values() for value in values if value is not None)
    total_bytes = sum(value for values in traffic.values() for value in values if value is not None)
    print("\n## Estimated operator cost (known-shape estimates only)")
    print("op                              nodes       FLOPs   FLOP %    bytes   bytes %")
    ranked = []
    for op in costs:
        flops = sum(value for value in costs[op] if value is not None)
        bytes_moved = sum(value for value in traffic[op] if value is not None)
        ranked.append((flops, op, bytes_moved))
    for flops, op, bytes_moved in sorted(ranked, reverse=True):
        flop_pct = 100 * flops / total_flops if total_flops else 0
        byte_pct = 100 * bytes_moved / total_bytes if total_bytes else 0
        print(f"{op:<30} {len(costs[op]):5d} {flops:12,d} {flop_pct:7.2f}% {bytes_moved:12,d} {byte_pct:7.2f}%")
    print("Note: symbolic dimensions and fused-kernel behavior make these upper-bound planning estimates, not measured runtime.")


def consumers(model: onnx.ModelProto) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            result[name].append(index)
    return result


def report_regions(model: onnx.ModelProto, shapes: dict[str, tuple[int | str, ...]]) -> None:
    regions = []
    start = None
    nodes_with_sentinel = list(model.graph.node) + [onnx.NodeProto()]
    for index, node in enumerate(nodes_with_sentinel):
        supported = node.op_type in ACCELERATOR_OPS
        if supported and start is None:
            start = index
        if not supported and start is not None:
            nodes = model.graph.node[start:index]
            inputs = sorted({name for item in nodes for name in item.input if name and name not in {out for n in nodes for out in n.output}})
            outputs = sorted({name for item in nodes for name in item.output if name and any(name in other.input for other in model.graph.node[index:])})
            regions.append((len(nodes), start, index - 1, nodes, inputs, outputs))
            start = None
    print("\n## Contiguous accelerator regions")
    for size, first, last, nodes, inputs, outputs in sorted(regions, reverse=True):
        print(f"{first:5d}-{last:<5d}  size={size:3d}  ops={Counter(node.op_type for node in nodes)}")
        print(f"       inputs:  {', '.join(f'{name}{shape_text(shapes.get(name))}' for name in inputs) or '[none]'}")
        print(f"       outputs: {', '.join(f'{name}{shape_text(shapes.get(name))}' for name in outputs) or '[graph-internal]'}")


def report_attention(model: onnx.ModelProto, shapes: dict[str, tuple[int | str, ...]]) -> None:
    consumers_by_name = consumers(model)
    producers = {output: i for i, item in enumerate(model.graph.node) for output in item.output}

    def upstream(name: str, allowed: set[str]) -> int | None:
        seen: set[str] = set()
        while name and name not in seen:
            seen.add(name)
            index = producers.get(name)
            if index is None:
                return None
            node = model.graph.node[index]
            if node.op_type not in allowed:
                return index
            name = node.input[0] if node.input else ""
        return None

    print("\n## Attention candidates")
    found = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Softmax":
            continue
        score_index = upstream(node.input[0] if node.input else "", {"Cast", "Add", "Div", "Mul"})
        if score_index is None or model.graph.node[score_index].op_type not in {"MatMul", "Einsum"}:
            continue
        score_node = model.graph.node[score_index]
        q_shape = shapes.get(score_node.input[0]) if score_node.input else None
        k_shape = shapes.get(score_node.input[1]) if len(score_node.input) > 1 else None
        following = consumers_by_name.get(node.output[0] if node.output else "", [])
        if following and model.graph.node[following[0]].op_type == "Cast":
            following = consumers_by_name.get(model.graph.node[following[0]].output[0], [])
        value_node = next(
            (model.graph.node[item] for item in following if model.graph.node[item].op_type in {"MatMul", "Einsum"}),
            None,
        )
        v_shape = shapes.get(value_node.input[1]) if value_node and len(value_node.input) > 1 else None
        score_shape = shapes.get(node.output[0])
        flops = matmul_flops(q_shape, k_shape) or 0
        if value_node:
            flops += matmul_flops(score_shape, v_shape) or 0
        if score_node.op_type == "Einsum" and score_shape and q_shape and k_shape:
            flops += (numeric_product(score_shape) or 0) * (q_shape[-1] if isinstance(q_shape[-1], int) else 0)
        if value_node and value_node.op_type == "Einsum" and value_node.output:
            value_shape = shapes.get(value_node.output[0])
            if value_shape and score_shape and v_shape and isinstance(v_shape[-1], int):
                flops += (numeric_product(value_shape) or 0) * v_shape[-1]
        found += 1
        print(f"softmax node {index}: Q={shape_text(q_shape)} K={shape_text(k_shape)} V={shape_text(v_shape)}")
        print(f"       softmax={shape_text(score_shape)} attention FLOPs={flops:,}" if flops else "       attention FLOPs=unknown")
    print(f"Attention candidates: {found} (heuristic: score MatMul/Einsum -> Softmax -> value MatMul/Einsum)")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRAPH
    model = onnx.load(str(path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = all_shapes(inferred)
    print(f"Graph: {path}")
    print(f"Nodes: {len(model.graph.node):,}; inputs: {len(model.graph.input):,}; outputs: {len(model.graph.output):,}")
    report_matmuls(inferred, shapes)
    report_einsums(inferred, shapes)
    report_costs(inferred, shapes)
    report_regions(inferred, shapes)
    report_attention(inferred, shapes)


if __name__ == "__main__":
    main()
