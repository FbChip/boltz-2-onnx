# Boltz-2 ONNX-Export Audit

Read-only walk of `./boltz/src/boltz/` against the v0 contract:
single-seq protein, `use_kernels=False`, steering disabled, **no** affinity head,
**yes** confidence head, **yes** full-atom output. Two graphs: `trunk.onnx`
(one trunk pass) + `diffusion_step.onnx` (one denoising step). All loops in JS.

File paths below are relative to `boltz/src/boltz/`.

## 1. Top-level forward & module split

- `Boltz2.forward()` at `model/models/boltz2.py:401`.
- Trunk → diffusion boundary at `boltz2.py:438–543`.
  - Trunk produces `s [B, N, token_s]`, `z [B, N, N, token_z]`, `pdistogram` logits.
  - `DiffusionConditioning` (`boltz2.py:516–522`) ingests `(s, z, distogram)` and
    emits 6 tensors `(q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias)`
    which feed every diffusion step. **New in Boltz-2** — not in Boltz-1.
- Recycling loop: Python `for` over `range(recycling_steps + 1)` at `boltz2.py:439–489`.
  Each iter adds recycle projections of previous `(s, z)` back into initial embeddings.
  → Export one trunk graph; JS owns the loop. Same shape as Boltz-1 plan.

## 2. Triangular attention path

- Pure-PyTorch fallback still exists: `model/layers/triangular_attention/primitives.py:156–280`
  (`Attention` class with `_chunk`).
- `use_kernels=False` correctly routes both `tri_att_start` and `tri_att_end`
  (`pairformer.py:91,99`) to the fallback.
- **Boltz-2 widens the kernel surface**: `TriangleMultiplicationIncoming/Outgoing`
  now also accept `use_kernels` (`pairformer.py:78,83`). Pure-PyTorch is the default
  branch when False — no new kernel-only paths.

## 3. Confidence head

- `model/modules/confidencev2.py:234–399`.
- Inputs: `s_inputs, s, z, x_pred, pred_distogram_logits, feats, multiplicity`.
- Outputs: `plddt_logits [B,N,50]` (or atom-level `[B,N,23,50]`),
  `pae_logits [B,N,N,64]`, `pde_logits [B,N,N,64]`, `resolved_logits [B,N,2]`.
- Aggregated scalars (pTM, ipTM, plddt means) computed in `confidencev2.py:325–351`
  — these can be moved JS-side if desired, but the logits are the load-bearing outputs.
- **Blocker**: `run_sequentially=True` triggers a Python loop over multiplicity
  (`confidencev2.py:124`). For export, always trace with `multiplicity=1,
  run_sequentially=False`.

## 4. Diffusion denoiser (one step)

- `model/modules/diffusionv2.py`. The exportable unit is
  `preconditioned_network_forward()` at `diffusionv2.py:251–274` — NOT `sample()`.
- Inputs per step: `(s_trunk, s_inputs, r_noisy, sigma, feats, diffusion_conditioning)`.
- Output: `denoised_coords` (same shape as `r_noisy`).
- Steering gated by three independent fields, all default `False`:
  `fk_steering`, `physical_guidance_update`, `contact_guidance_update`. Setting
  all three False removes every steering branch via Python-time constant folding.
- Coordinate rotation einsums at `diffusionv2.py:356,361,368`:
  `"bmd,bds->bms"` — trivial, will export cleanly.

## 5. Suspicious patterns for `torch.onnx.export(..., dynamo=True)`

| Pattern | Where | Severity |
|---|---|---|
| `.item()` on sigmas, then Python arithmetic | `diffusionv2.py:372` (`t_hat = sigma_tm * (1+gamma)`) | **High** — values bake in at trace time |
| `.item()` on mask sums, then dynamic `pad()` | `confidencev2.py:368,384`; `layers/confidence_utils.py` | **High** — dynamic shape from data |
| Python loop over `range(multiplicity)` | `confidencev2.py:124` | Medium — trace with multiplicity=1 |
| `chunk()` count from `multiplicity % max_parallel` | `diffusionv2.py:383–385` | Sampler-loop only (not in step graph) |
| `torch.multinomial` with dynamic shape | `diffusionv2.py:482–494` | FK-steering only; disabled in v0 |
| `@torch.compiler.disable` on tri-mult & attn LN | `triangular_mult.py:7`, `primitives.py:199` | Low — just compile gates, not export-hostile |
| Non-trivial einsums | `triangular_mult.py` (`bikd,bjkd->bijd`), `outer_product_mean.py` (`bsic,bsjd->bijcd`), `pair_averaging.py` | Low — dynamo handles these in recent torch; rewrite as `matmul + permute` only if a specific one breaks |

## 6. Boltz-2 modules not present in Boltz-1

| Module | In v0 graph? | Notes |
|---|---|---|
| `modules/affinity.py` | **No** | Out of scope |
| `modules/confidencev2.py` | Trunk-side | New v2 confidence (vs Boltz-1 v1) |
| `modules/diffusionv2.py` | Diffusion-step | New atom-level v2 denoiser |
| `modules/encodersv2.py` | Diffusion-step | Atom & token encoder; uses window-bucketed indexing |
| `modules/transformersv2.py` | Diffusion-step | Diffusion token transformer |
| `modules/trunkv2.py` | Trunk-side | Contains `ContactConditioning`, `DistogramModule`, `BFactorModule` |
| `modules/diffusion_conditioning.py` | Trunk → diffusion bridge | Produces 6 bias/query tensors consumed every step |

## Top-3 risks (and tractability)

1. **`diffusionv2.py:372` — `.item()` on sigmas.** Fix: in the export wrapper,
   pass `sigma_tm, sigma_t, gamma` as 0-d tensors through to the network and
   eliminate the Python-level `t_hat = sigma_tm * (1 + gamma)` by inlining tensor
   ops. Local, mechanical.
2. **`confidencev2.py:368,384` — dynamic mask-driven padding.** Fix: declare the
   atom dimension as a dynamic axis in `dynamic_shapes` and pass pre-padded
   inputs. Pad-on-demand removed from the export-only wrapper.
3. **`confidencev2.py:124` — multiplicity loop.** Fix: trace with
   `multiplicity=1, run_sequentially=False`. Export sees a single forward pass.

None of these are research; all are local rewrites in our export wrapper.
Triangle attention is *not* on the blocker list — the pure-PyTorch fallback
exports cleanly and matches the trifast kernel numerically.
