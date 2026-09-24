# Operator acceleration summary

The percentages below are static estimated FLOP shares from inferred tensor
shapes. They are not measured wall-clock runtime percentages. Memory is a
qualitative assessment based on tensor traffic and layout pressure.

## `fp16/trunk_fp16.onnx`

| Operator | FLOPs | Memory | Accelerator? |
|---|---:|---|---|
| MatMul | 99.82% | High | YES |
| LayerNorm | 0.06% | Medium | YES |
| Softmax | 0.02% | Medium | YES |
| Einsum | 0.01% | High | MAYBE |
| Everything else | 0.10% | Low/Medium | CPU or fused |

## `fp16/diffusion_step_fp16.onnx`

| Operator | FLOPs | Memory | Accelerator? |
|---|---:|---|---|
| MatMul | 99.67% | High | YES |
| LayerNorm | 0.09% | Medium | YES |
| Softmax | 0.03% | Medium | YES |
| Einsum | 0.03% | High | MAYBE |
| Everything else | 0.18% | Low/Medium | CPU or fused |

## Compiler interpretation

- **MatMul:** first-generation accelerator priority; it dominates arithmetic in both graphs.
- **LayerNorm:** implement as a fused accelerator kernel with scale/bias.
- **Softmax:** implement as a numerically stable fused attention kernel where possible.
- **Einsum:** lower recognized attention equations to batched MatMul; retain a fallback for the remaining tensor contractions.
- **Everything else:** CPU fallback is acceptable for a first demo, but Cast, Reshape, Transpose, Add, and Mul should be fused or kept device-local to avoid transfer overhead.

The “Accelerator?” column describes a proposed partition, not an existing
compiled backend implementation.
