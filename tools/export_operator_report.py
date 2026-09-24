"""Export MatMul, Einsum, and Softmax shape/FLOP reports for fp16 graphs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

import onnx


ROOT = Path(__file__).resolve().parents[1]
GRAPHS = (
    ROOT / "fp16" / "trunk_fp16.onnx",
    ROOT / "fp16" / "diffusion_step_fp16.onnx",
)
OPERATORS = {"MatMul", "Einsum", "Softmax"}


def shape_of(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value or dim.dim_param or "?")
    return tuple(shape)


def shapes_for(model: onnx.ModelProto) -> dict[str, tuple[int | str, ...]]:
    result = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        result[value.name] = shape_of(value)
    for initializer in model.graph.initializer:
        result.setdefault(initializer.name, tuple(initializer.dims))
    return result


def product(shape: Iterable[int | str] | None) -> int | None:
    if shape is None:
        return None
    total = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            return None
        total *= dimension
    return total


def matmul_flops(left: tuple[int | str, ...] | None, right: tuple[int | str, ...] | None) -> int | None:
    if not left or not right or len(left) < 2 or len(right) < 2:
        return None
    m, k_left, k_right, n = left[-2], left[-1], right[-2], right[-1]
    if not all(isinstance(value, int) for value in (m, k_left, k_right, n)) or k_left != k_right:
        return None
    left_batch = product(left[:-2]) or 1
    right_batch = product(right[:-2]) or 1
    return 2 * left_batch * right_batch * m * n * k_left


def einsum_flops(equation: str, input_shapes: list[tuple[int | str, ...] | None]) -> int | None:
    equation = equation.replace(" ", "")
    left, separator, right = equation.partition("->")
    if not separator or "..." in equation:
        return None
    terms = left.split(",")
    if len(terms) != len(input_shapes):
        return None
    dimensions: dict[str, int] = {}
    for term, shape in zip(terms, input_shapes):
        if shape is None or len(term) != len(shape):
            return None
        for label, dimension in zip(term, shape):
            if not isinstance(dimension, int):
                return None
            if label in dimensions and dimensions[label] != dimension:
                return None
            dimensions[label] = dimension
    total = product(dimensions.values())
    if total is None:
        return None
    return 2 * total


def shape_text(shape: tuple[int | str, ...] | None) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]" if shape is not None else "[unknown]"


def report_for(path: Path) -> list[dict[str, str]]:
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path), load_external_data=False))
    shapes = shapes_for(model)
    rows = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in OPERATORS:
            continue
        input_shapes = [shapes.get(name) for name in node.input if name]
        output_shapes = [shapes.get(name) for name in node.output if name]
        if node.op_type == "MatMul":
            flops = matmul_flops(input_shapes[0], input_shapes[1])
        elif node.op_type == "Einsum":
            equation = next((attribute.s.decode() for attribute in node.attribute if attribute.name == "equation"), "")
            flops = einsum_flops(equation, input_shapes)
        else:
            output_elements = product(output_shapes[0] if output_shapes else None)
            flops = output_elements * 5 if output_elements is not None else None
        rows.append({
            "Graph": path.name,
            "Node Index": str(index),
            "Operator": node.op_type,
            "Tensor Shapes": "inputs=" + ", ".join(shape_text(shape) for shape in input_shapes)
            + " -> outputs=" + ", ".join(shape_text(shape) for shape in output_shapes),
            "Estimated FLOPs": str(flops) if flops is not None else "unknown",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Graph", "Node Index", "Operator", "Tensor Shapes", "Estimated FLOPs"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    selected = [Path(argument) for argument in sys.argv[1:]] or list(GRAPHS)
    all_rows = []
    for graph in selected:
        rows = report_for(graph)
        output = graph.with_name(graph.stem + "_operator_report.csv")
        write_csv(output, rows)
        all_rows.extend(rows)
        counts = {}
        for row in rows:
            counts[row["Operator"]] = counts.get(row["Operator"], 0) + 1
        print(f"{graph.name}: {len(rows):,} rows ({counts}); wrote {output}")
    if len(selected) > 1:
        combined = ROOT / "fp16_operator_report.csv"
        write_csv(combined, all_rows)
        print(f"Combined report: {len(all_rows):,} rows; wrote {combined}")


if __name__ == "__main__":
    main()
