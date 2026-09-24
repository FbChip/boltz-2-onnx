---
license: mit
tags:
  - protein-structure-prediction
  - boltz
  - alphafold
  - onnx
  - webgpu
  - biology
library_name: onnx
base_model: boltz-community/boltz-2
---

# Boltz-2 ONNX (single-sequence v0)

ONNX-Runtime-compatible export of **Boltz-2** ([Wohlwend et al., 2024–2025](https://doi.org/10.1101/2025.06.14.659707); MIT-licensed), produced by [boltz-dev](https://github.com/jwohlwend/boltz). Split into two graphs so the inference loop (recycling + diffusion sampling) can run client-side — designed for [biocircus.io](https://biocircus.io), which loads these via ONNX Runtime Web + WebGPU and lets a user predict a structure entirely inside the browser tab.

[![Open fp16 graph viewer in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FbChip/boltz-2-onnx/blob/main/colab_fp16_graph.ipynb)

The [Colab notebook](colab_fp16_graph.ipynb) downloads the fp16 graphs, reports their signatures, detects CUDA support, and opens the graphs in an interactive Netron viewer. A complete sequence-to-structure run additionally needs the Boltz preprocessing and orchestration pipeline.

For compiler investigation, `tools/analyze_diffusion_graph.py` inventories MatMul shapes and FLOPs, expands Einsum equations, estimates operator FLOPs and memory traffic, reports contiguous accelerator regions for MatMul/LayerNormalization/Softmax, and identifies attention candidates:

```bash
python tools/analyze_diffusion_graph.py
```

The cost and traffic values are static planning estimates based on inferred shapes, not measured runtime.

This is **v0**: single-sequence protein only, no MSA, no templates, no affinity head. Confidence and full-atom output included. Three precision tiers.

## What's here

| Tier  | Trunk      | Diffusion step | Total       | When to use                          |
|-------|------------|----------------|-------------|--------------------------------------|
| fp32  | 795 MB     | 1.05 GB        | 1.85 GB     | Reference / debug                    |
| fp16  | 409 MB     | 538 MB         | **947 MB**  | Desktop / laptop default — recommended |
| int8  | 217 MB     | 273 MB         | **490 MB**  | Smartphone / tablet / low-RAM        |

```
fp32/                          fp16/                          int8/
├── trunk.onnx                 ├── trunk_fp16.onnx            ├── trunk_int8.onnx
├── trunk.onnx.data            ├── trunk_fp16.onnx.data       ├── trunk_int8.onnx.data
├── diffusion_step.onnx        ├── diffusion_step_fp16.onnx   ├── diffusion_step_int8.onnx
└── diffusion_step.onnx.data   └── diffusion_step_fp16.onnx.data └── diffusion_step_int8.onnx.data
```

Each `.onnx` file is paired with an external-data sidecar (`.onnx.data`) holding the bulk of the weights. ONNX Runtime loads the pair automatically when both are co-located.

## Graph signatures

### Trunk (one recycling pass)

**Inputs** (80 total): 78 preprocessed feature tensors from the Boltz-2 input pipeline — `token_pad_mask`, `atom_pad_mask`, `ref_pos`, `ref_element`, `res_type`, `residue_index`, `token_bonds`, `type_bonds`, `mol_type`, `entity_id`, `sym_id`, etc. — plus `s_prev [B, N, 384]` and `z_prev [B, N, N, 128]` (zeros on the first recycling iteration; trunk outputs fed back on subsequent ones).

**Outputs** (9 total):
- `s [B, N, 384]` — single representation
- `z [B, N, N, 128]` — pair representation
- `pdistogram [B, N, N, 1, 64]` — distogram logits
- `q [B, A, 128]`, `c [B, A, 128]` — atom-level diffusion conditioning
- `atom_enc_bias [B, K, W, H, 12]`, `atom_dec_bias [B, K, W, H, 12]`, `token_trans_bias [B, N, N, 384]` — diffusion biases
- `s_inputs [B, N, 384]` — input embedder output, reused by the diffusion step

### Diffusion step (one denoising iteration)

**Inputs** (87 total): the same 78 feature tensors plus the 8 trunk-cached tensors (`s`, `s_inputs`, `q`, `c`, three biases) + `x_noisy [B, A, 3]` (current atom coords) + `sigma [B]` (1-D tensor; the `t_hat = sigma_tm * (1 + gamma)` value computed in the orchestrator).

**Output**: `x_denoised [B, A, 3]`.

`to_keys` (used by the atom encoder) is **not** part of the graph contract — it's a `functools.partial` closure in PyTorch land. Reconstruct it inside the diffusion graph by computing `get_indexing_matrix(K = A_padded / W, W, H, device)` with the known model hyperparams W=32, H=128 (and equivalently for any other atom-window split).

## Validation

Per the [Phase 5 report](reports/phase5_validation.md), all three precisions sit *inside* the natural PyTorch inter-seed noise distribution for 1CRN single-sequence inference. Pairwise Kabsch-aligned Cα RMSD (Å):

```
                  PyTorch ⟷ PyTorch       ORT ⟷ PyTorch
  inter-seed pairs:  mean 6.77, range 5.69–8.06
  fp32  vs  PyTorch: mean 6.74, range 4.94–8.99
  fp16  vs  PyTorch: mean 6.73, range 4.95–8.97
  int8  vs  PyTorch: mean 6.20, range 4.97–8.01

  Cross-precision (ORT vs ORT, same seed):
  fp32  ↔  fp16:  0.186   ← essentially lossless
  fp32  ↔  int8:  2.31    ← within ORT cross-augmentation drift
  fp16  ↔  int8:  2.27
```

The high absolute RMSDs are intrinsic to **single-sequence-no-MSA** Boltz-2 (pLDDT ~0.45 on this prediction — the model is sampling broadly because it has no co-evolution signal). With MSA inputs the noise floor collapses ~10×; quantisation quality would still hold.

## Use in biocircus.io

The biocircus runtime fetches a `ModelManifest` that points at this repo, picks a precision tier based on hardware probe, and runs the recycling + diffusion sampling loops in TypeScript against ORT Web sessions. See [boltz_orchestrate.py](https://github.com/biocircus/boltz-dev/blob/main/scripts/boltz_orchestrate.py) for a Python reference implementation of the orchestration loop.

For the per-step diffusion math the orchestrator needs:
- A Haar-uniform random-rotation generator (called `compute_random_augmentation` in Python).
- A Kabsch-style weighted rigid align (used when `alignment_reverse_diff=True`, which it is by default in Boltz-2).
- The Karras noise schedule (formula and constants in `phase4/validation_report.md`).

## License & citation

MIT, matching the upstream Boltz repo. Please cite Boltz-2:

```bibtex
@article{wohlwend2025boltz2,
  title  = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  author = {Wohlwend, J. and others},
  year   = {2025},
  doi    = {10.1101/2025.06.14.659707}
}
```

## Provenance

Exported with `torch.onnx.export(..., dynamo=True)` on PyTorch 2.12, ONNX opset 18. Quantised with `onnxconverter_common.float16` (fp16) and `onnxruntime.quantization.quantize_dynamic` (int8). See the [boltz-dev](https://github.com/jwohlwend/boltz) repo's `scripts/` directory for the export and quantisation scripts, and the EXPORT_PLAN.md pitfall catalogue for documented `onnxconverter_common` workarounds on dynamo-exported graphs.
