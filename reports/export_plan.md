# Boltz → ONNX Export Plan

A phased plan. Each phase is a checkpoint: you can stop here and the work so far is still load-bearing for whatever comes next. **Don't skip ahead** — Phase 2 depends on Phase 1 working in a specific way.

The contract you're producing against lives at **`../biocircus/docs/BOLTZ_ONNX_SPEC.md`**. Read it before Phase 0.

---

## Phase 0 — Environment

**Goal:** a reproducible Python environment where Boltz runs end-to-end in `--no_kernels` mode on a small target.

Steps:

1. Pick a package manager. `uv` is fastest; `poetry` is fine; plain venv + pip works.
2. Python 3.11 or 3.12. (Boltz's setup may pin specific versions — defer to its `pyproject.toml`.)
3. Install Boltz from upstream:
   ```
   pip install boltz
   # or, for the dev/HEAD path:
   git clone https://github.com/jwohlwend/boltz.git && pip install -e ./boltz
   ```
4. Install ONNX tooling:
   ```
   pip install onnx onnxruntime onnxscript
   ```
5. Smoke test:
   ```
   boltz predict --use_msa_server false --no_kernels --num_samples 1 \
       --recycling_steps 1 --sampling_steps 50 --output_format pdb \
       <path-to-1CRN.fasta>
   ```
   If this completes and produces a plausible Cα ribbon, the environment is good.

Checkpoint: a CLI run that takes < 5 minutes on CPU for 1CRN (46 residues). Save the output as `phase0_reference/1CRN_baseline.pdb`. This is your golden output — every subsequent export must match it within tolerance.

---

## Phase 1 — Anatomy

**Goal:** know exactly what code path `Boltz1.forward()` traces with `use_kernels=False, steering disabled, single-sequence`.

Don't write export code yet. Instrument and read.

Steps:

1. Add a hook at the top of `Boltz1.forward()` (monkey-patch from a wrapper script, don't edit Boltz in-place) that logs:
   - shapes of all incoming tensors
   - the value of every `if` branch taken
   - shapes at the trunk/diffusion boundary
2. Run the Phase 0 smoke test under the hook. Save the trace to `phase1_trace/forward_trace.txt`.
3. Identify the exact tensors that cross the trunk → diffusion boundary. The contract spec calls these `s` (single representation) and `z` (pair representation). Confirm shapes and dtypes match what `meta.json` will declare (`d_single`, `d_pair`).
4. Identify the recycling boundary: what does `s_prev` look like on iteration 0 (likely zeros) and what shape/dtype must it be?
5. Map every Python `if` in the diffusion sampler. Confirm each one folds to a constant under `steering_args` disabled. If any depends on the diffusion step index, it has to be lifted out of the graph.

Checkpoint: a written one-pager `phase1_trace/anatomy.md` that lists, for both graphs, every input tensor + shape + dtype + dependence on (sequence length, recycling step, diffusion step). This is the source of truth for what you'll export.

---

## Phase 2 — Trunk export

**Goal:** `trunk.onnx` that, given `(token_ids, attention_mask, residue_index, s_prev, z_prev)`, returns `(s, z, distogram_logits)`.

Steps:

1. Write a thin wrapper module `boltz_trunk_module.py` whose `forward()` calls into Boltz's trunk only, with `use_kernels=False`. Don't include the recycling loop — that's JS-side. One pass only.
2. Trace and export:
   ```python
   torch.onnx.export(
       wrapper_module,
       example_inputs,
       'trunk.onnx',
       opset_version=18,
       dynamo=True,  # newer exporter handles modern ops better
       input_names=['token_ids', 'attention_mask', 'residue_index', 's_prev', 'z_prev'],
       output_names=['s', 'z', 'distogram_logits'],
       dynamic_shapes={...},  # L as symbolic
   )
   ```
3. Validate the graph loads in `onnxruntime` Python:
   ```python
   sess = ort.InferenceSession('trunk.onnx', providers=['CPUExecutionProvider'])
   out = sess.run(None, example_feeds)
   ```
4. Compare outputs to a direct PyTorch call on the same inputs. Max abs diff on `s` and `z` should be < 1e-3 fp32, < 1e-2 fp16.

Likely pitfalls:
- **`scaled_dot_product_attention`** with a non-trivial mask shape may not export cleanly. The pure-PyTorch `_attention()` path in `primitives.py` is plain matmul + softmax + matmul — that exports cleanly. Make sure the wrapper monkey-patches the attention impl to that path.
- **Dynamic control flow** inside Pairformer. Confirm none of the loops are data-dependent.
- **`torch.einsum`** with complex patterns sometimes traces oddly. If a specific einsum breaks, rewrite it as explicit `matmul + permute` calls in a fork.
- **Memory blow-up** on export. The pair representation is `[1, L, L, 128]` — for L=400 that's already huge. Export with L=64 or L=128 example shapes; mark L as dynamic.

Checkpoint: `trunk.onnx` plus a validation script that asserts < tolerance vs PyTorch reference. Both checked into the repo.

---

## Phase 3 — Diffusion-step export

**Goal:** `diffusion_step.onnx` that, given `(s, z, x_noisy, sigma)`, returns `x_denoised`. **One step only** — not the loop.

Steps:

1. Write `boltz_diffusion_step_module.py` that wraps Boltz's denoising network with steering disabled. Take the noise level `sigma` as an input tensor, not a precomputed constant baked into the graph.
2. Export with opset 18, dynamic L.
3. Validate against a single PyTorch denoising step on the same `(s, z, x_noisy, sigma)`.

Checkpoint: `diffusion_step.onnx` that passes a per-step Cα-coordinate diff under tolerance (1e-3 fp32) versus a PyTorch denoising step.

---

## Phase 4 — End-to-end Python validation

**Goal:** orchestrate the recycling + diffusion loops in Python using only ONNX Runtime, and match the Phase 0 baseline within tolerance.

Steps:

1. Write `boltz_orchestrate.py`. Loads both ONNX graphs into ORT sessions.
2. Implement the recycling loop:
   - Start with `s_prev = zeros`, `z_prev = zeros`.
   - For `i in range(recycling_steps + 1)`: run `trunk.onnx`, feed outputs back as next iteration's `s_prev`, `z_prev`.
3. Implement the diffusion sampler:
   - Read the noise schedule from `meta.json` (which you also produce in this phase).
   - For each diffusion step: build the noisy coords (or use previous step's denoised + noise increment), call `diffusion_step.onnx`, advance.
4. Write the final Cα coords as PDB or mmCIF.
5. Run on 1CRN (46 aa) and 1UBQ (76 aa). Compare Cα RMSD against `phase0_reference/*.pdb`. Target: < 0.5 Å.

Checkpoint: `validation_report.md` showing the RMSD numbers. If both targets pass, the artifacts are shippable.

---

## Phase 5 — Quantisation and packaging

**Goal:** fp16 weights, organised as a single uploadable bundle.

Steps:

1. Quantise both graphs to fp16:
   ```python
   from onnxconverter_common import float16
   model_fp16 = float16.convert_float_to_float16(onnx.load('trunk.onnx'))
   ```
2. Re-validate at fp16 against the same 1CRN / 1UBQ targets. If RMSD jumps above 0.5 Å, debug the offending op (some softmax / norm layers prefer fp32 — keep those in fp32 and convert the rest).
3. Write the final `meta.json` with all hyperparameters, the noise schedule array, and the architecture constants the TS side needs.
4. Package: `boltz1-onnx-v0/{trunk.onnx, diffusion_step.onnx, meta.json}`.
5. Upload to `huggingface.co/biocircus/boltz1-onnx-v0` (or wherever the user designates). Make the repo public so cross-origin fetches work without auth.

Checkpoint: a URL the biocircus side can drop into a `ModelManifest` and load.

---

## Phase 6 — Handoff

Open a draft PR on the biocircus repo (sibling directory). Include:
- The artifact URLs.
- Any spec deltas you needed (and why). If shapes diverged, the contract gets a v0.2 bump.
- The validation_report.md.
- A `boltz_orchestrate.py`-style reference implementation in Python that the JS orchestration loop can mirror.

The biocircus side will wire `useModelSession` against the manifest, render the structure in the existing Mol\* canvas, and the disabled "Predict structure" button finally lights up.

---

## Pitfall catalogue (collect as you find them)

Live document; append as you discover things. Each entry: symptom → root cause → fix.

### P-1 — `to_keys` is a `functools.partial`, not a tensor
- **Symptom:** `AttributeError: 'functools.partial' object has no attribute 'shape'`
  in any code that inspects trunk outputs. The 3rd output of
  `DiffusionConditioning.forward` is a Python callable, not a tensor.
- **Root cause:** `AtomEncoder.forward` (`encodersv2.py:350`) constructs
  `to_keys = partial(single_to_keys, indexing_matrix=…, W=W, H=H)`. The
  closure captures `keys_indexing_matrix` (a tensor) plus two ints. It is
  passed through `DiffusionConditioning` unchanged and is reused inside the
  diffusion sampler loop.
- **Fix (trunk side):** discard `_to_keys` in the trunk wrapper. The matrix
  is deterministic from `(K, W, H, device)` where `K = padded_atom_count /
  atoms_per_window_queries`, so the diffusion graph can recompute it
  internally with the same `get_indexing_matrix(K, W, H, device)` call.

### P-2 — `RelativePositionEncoder.cyclic_pos_enc` triggers a data-dependent guard
- **Symptom:** `torch.export.export` raises
  `GuardOnDataDependentSymNode: Could not guard on Eq(u0, 1)` pointing at
  `encodersv2.py:64`:
  ```python
  if self.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0):
  ```
  Even with `cyclic_period` all-zero, `torch.any(...)` is a tensor reduction
  whose Python-bool result is opaque to the tracer.
- **Root cause:** the runtime conditional creates a data-dependent branch
  that the exporter cannot symbolically resolve.
- **Fix:** set `model.rel_pos.cyclic_pos_enc = False` on the live module
  before export. Python's `and` short-circuits, the `torch.any` call never
  runs, and dynamo sees only the fall-through path. Acceptable for v0 since
  cyclic peptides aren't in scope.

### P-3 — Slight numerical drift on the trunk's `s` output (fp32)
- **Symptom:** validation against PyTorch reports `max_abs_diff ≈ 1.04e-3`
  on the `s` (single-representation) output. Other outputs (`z`,
  `pdistogram`, `q`, `c`, three biases) sit comfortably below 3e-4.
- **Root cause:** dynamo's ONNX optimizer fuses linears in an order that
  differs from PyTorch's, and the accumulated rounding shows up on the
  longest-chained tensor in the graph (48 Pairformer blocks + trunkv2
  preludes).
- **Fix:** none required for v0 — within fp32 noise at this depth, and JS
  inference will run fp16 where the tolerance window is `< 1e-2`. Revisit
  only if a downstream consumer flags it.

### P-4 — `onnxconverter_common.float16` misses Cast.to attributes
- **Symptom:** ORT session creation fails with `Type Error: Type
  (tensor(float16)) of output arg (...) of node (...) does not match
  expected type (tensor(float))` on a `_to_copy` / `convert_element_type` /
  Cast-like node.
- **Root cause:** the converter updates initializers and op-level dtype
  declarations but does NOT update `Cast.to` attribute values from
  FLOAT(=1) to FLOAT16(=10). Dynamo-exported graphs are Cast-heavy: 2756
  on the trunk, 293 on the diffusion step.
- **Fix:** post-process the converted model — for every Cast node with
  `to == 1`, set `to = 10`. Implementation in `_patch_node_attrs_to_fp16`
  in `scripts/quantize.py`.

### P-5 — `ConstantOfShape` with no `value` attribute defaults to fp32
- **Symptom:** ORT loader Type Error pointing at a ConstantOfShape's output
  that's declared fp16 but consumed as fp32.
- **Root cause:** ONNX spec: ConstantOfShape with no `value` attribute
  outputs FLOAT-zero. The converter doesn't *add* a value attribute when
  one is missing — so it stays fp32 by default.
- **Fix:** during the post-process patch, inject an explicit fp16-zero
  `value` attribute on any bare ConstantOfShape. Three such nodes on the
  trunk.

### P-6 — `RandomUniformLike` from no-op dropout has no fp16 CPU kernel
- **Symptom:** ORT loader fails with
  `NOT_IMPLEMENTED: Could not find an implementation for RandomUniformLike`.
- **Root cause:** Boltz's `get_dropout_mask` computes
  `torch.rand(...) >= 0` even in eval mode (the math is structurally a
  tensor-of-ones because eval has `dropout * False = 0`). dynamo bakes the
  rand call into the graph as a RandomUniformLike. ORT's CPU EP lacks an
  fp16 kernel for it. 284 such nodes on the trunk.
- **Fix:** replace every `RandomUniformLike` / `RandomNormalLike` with
  `Identity`. The input is a ConstantOfShape-zeros tensor; `Identity(zeros)
  >= 0` still produces the all-ones mask the original math gave us.

### P-7 — `value_info` annotations not retyped by the fp16 converter
- **Symptom:** ORT loader Type Errors of the form "Type (tensor(float16))
  of output arg X does not match expected type (tensor(float))" where X is
  a graph-internal tensor.
- **Root cause:** `model.graph.value_info` entries store
  intermediate-tensor type annotations from the original fp32 export.
  onnxconverter_common doesn't rewrite them. ORT trusts those annotations
  and faults on conflicts with the now-fp16 actual node outputs.
- **Fix:** `del fp16_model.graph.value_info[:]` after conversion. ORT
  re-infers cleanly.

### P-8 — int8 dynamic quantizer's shape-infer pass trips on dynamo graphs
- **Symptom:** `onnxruntime.quantization.quantize_dynamic` fails with
  `InferenceError: Inferred shape and existing shape differ in dimension 0:
  (1) vs (128)`.
- **Root cause:** dynamo-exported graphs ship `value_info` entries whose
  shapes conflict with what onnx shape inference (run inside the
  quantizer) produces. The quantizer treats this as fatal.
- **Fix:** same as P-7 — strip `value_info` before invoking
  `quantize_dynamic`. The quantizer re-infers shapes successfully.

---

## What not to do

- **Don't run inference in this project.** That's biocircus's job. Validation = "ONNX matches PyTorch", not "the prediction is good." The latter is biocircus's integration test.
- **Don't add a CLI.** This project produces artifacts. The user does not need a tool to invoke them locally — they have biocircus for that.
- **Don't fork Boltz.** Import, monkey-patch in your own scripts, save patches as Python files in this repo. Upstream stays clean.
- **Don't bake recycling or diffusion loops into ONNX.** The whole point of the two-graph split is per-step control + progress reporting on the JS side.
- **Don't quantise to int8 without explicit user approval.** We learned the hard way that int8 ONNX exports can silently lose output calibration (see [[xenova-quantized-esm2-broken]] in biocircus memory). fp16 is the safe default.
