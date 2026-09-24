"""Group and rank MatMul shapes in the fp16 trunk and diffusion graphs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[1]
GRAPHS = (
    ROOT / "fp16" / "trunk_fp16.onnx",
    ROOT / "fp16" / "diffusion_step_fp16.onnx",
)


def shape_of(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    return tuple(dim.dim_value or dim.dim_param or "?" for dim in value_info.type.tensor_type.shape.dim)


def shape_map(model: onnx.ModelProto) -> dict[str, tuple[int | str, ...]]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    result = {value.name: shape_of(value) for value in values}
    result.update({item.name: tuple(item.dims) for item in model.graph.initializer})
    return result


def product(shape: tuple[int | str, ...]) -> int | None:
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        total *= dim
    return total


def flops(left: tuple[int | str, ...], right: tuple[int | str, ...]) -> int | None:
    if len(left) < 2 or len(right) < 2:
        return None
    m, k_left, k_right, n = left[-2], left[-1], right[-2], right[-1]
    if not all(isinstance(value, int) for value in (m, k_left, k_right, n)) or k_left != k_right:
        return None
    return 2 * (product(left[:-2]) or 1) * (product(right[:-2]) or 1) * m * n * k_left


def shape_text(shape: tuple[int | str, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def grouped_rows(path: Path) -> list[dict[str, str]]:
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path), load_external_data=False))
    shapes = shape_map(model)
    groups: Counter[tuple[str, str, str, str]] = Counter()
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        left = shapes.get(node.input[0], ("unknown",))
        right = shapes.get(node.input[1], ("unknown",))
        output = shapes.get(node.output[0], ("unknown",))
        estimate = flops(left, right)
        groups[(shape_text(left), shape_text(right), shape_text(output), str(estimate) if estimate is not None else "unknown")] += 1
    rows = [
        {
            "Graph": path.name,
            "Input A": key[0],
            "Input B": key[1],
            "Output": key[2],
            "FLOPs per occurrence": key[3],
            "Occurrences": str(count),
        }
        for key, count in groups.items()
    ]
    rows.sort(key=lambda row: float(row["FLOPs per occurrence"]) if row["FLOPs per occurrence"].isdigit() else -1, reverse=True)
    return rows


def main() -> None:
    markdown = ["# Grouped MatMul report", "", "Sorted by FLOPs per occurrence descending.", ""]
    for path in GRAPHS:
        rows = grouped_rows(path)
        output = path.with_name(path.stem + "_matmul_groups.csv")
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Graph", "Input A", "Input B", "Output", "FLOPs per occurrence", "Occurrences"])
            writer.writeheader()
            writer.writerows(rows)
        markdown.extend([f"## `{path.name}`", "", "| Input A | Input B | Output | FLOPs per occurrence | Occurrences |", "|---|---|---|---:|---:|"])
        for row in rows:
            markdown.append(f"| `{row['Input A']}` | `{row['Input B']}` | `{row['Output']}` | {row['FLOPs per occurrence']} | {row['Occurrences']} |")
        markdown.extend(["", f"Grouped shape signatures: **{len(rows)}**", ""])
        print(f"{path.name}: {len(rows)} grouped signatures; wrote {output}")
    (ROOT / "reports" / "matmul_groups.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {ROOT / 'reports' / 'matmul_groups.md'}")


if __name__ == "__main__":
    main()
