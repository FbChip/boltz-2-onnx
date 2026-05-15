# Boltz-2 ONNX — single-sequence v0 — completed handoff

**Status:** all three precision tiers (fp32, fp16, int8) of the three required graphs (trunk, diffusion step, confidence head) are exported, validated, and published to **[`latentspacecraft/boltz-2-onnx`](https://huggingface.co/latentspacecraft/boltz-2-onnx)** on Hugging Face. The Python orchestrator (`scripts/boltz_orchestrate.py`) drives an end-to-end prediction on 1CRN using only the ONNX graphs and produces a structurally valid Cα backbone matching PyTorch within natural seed-noise. User has visually confirmed the resulting fold.

**Audience:** the biocircus repo's main coding agent (operating in `../biocircus/`). Read this end-to-end before wiring up the TS side.

---

## What's on Hugging Face

```
latentspacecraft/boltz-2-onnx/
├── README.md                                 (model card)
├── meta.json                                 (everything you need to drive the graphs)
├── fp32/   trunk.onnx + .data,  diffusion_step.onnx + .data,  confidence.onnx + .data
├── fp16/   trunk_fp16.onnx + .data,           …                  …
├── int8/   trunk_int8.onnx + .data,           …                  …
└── reports/
    ├── COMPLETED.md                          (this file)
    ├── export_plan.md                        (full pitfall catalogue P-1..P-9)
    ├── phase4_validation.md                  (e2e noise-floor + fp32 numerics)
    └── phase5_validation.md                  (per-precision geometry + RMSDs)
```

Sizes (per precision, all three graphs combined):

| Precision | Total | Notes |
|---|---|---|
| fp32 | ~1.95 GB | Reference / debug |
| **fp16** | **~1.00 GB** | Default desktop/laptop tier — visually lossless vs fp32 |
| **int8** | **~520 MB** | Smartphone OPFS tier — bond-length geometry slightly compressed but globally correct |

The `.onnx.data` sidecars are Xet-stored on HF and resolve automatically when the matching `.onnx` is fetched into the same directory. ONNX Runtime requires both files to live side-by-side.

---

## What v0 includes and excludes

**Includes:**
- Single-sequence protein prediction (no MSA, no templates).
- Three ONNX graphs: `trunk`, `diffusion_step`, `confidence`.
- Per-residue pLDDT logits → biocircus can colour the Mol* ribbon by pLDDT.
- Per-pair PAE / PDE logits and per-residue resolved logits if you want to expose them later.

**Excludes (deferred to a v0.1 or v1 export):**
- MSA-fed inference (would need a new trunk export with the MSA path enabled).
- Affinity head (Boltz-2 specific, ligand binding affinity).
- pTM / ipTM scalars — straightforward to recompute on the TS side from `pae_logits` using the same formula in `boltz/model/loss/confidence.py`, but not wired into the graph.
- Dynamic shapes — current graphs are exported with concrete `N=46, A=352` (1CRN-sized). Any L ≠ 46 needs a re-export with `dynamic_shapes={…}` in `torch.onnx.export`; not blocking for the initial demo, but blocking for "type any sequence."

---

## The contract — `meta.json`

`meta.json` (16 KB, in the repo root on HF) is the authoritative spec. Fetch it first; it contains:

- **Schema version** + model identity.
- **Diffusion constants** the sampling loop needs: `sigma_min`, `sigma_max`, `sigma_data`, `rho`, `gamma_0`, `gamma_min`, `noise_scale`, `step_scale`, `alignment_reverse_diff`, default `num_sampling_steps`, plus the Karras schedule formula in human-readable form.
- **Model hyperparams**: `token_s=384`, `token_z=128`, `atoms_per_window_queries_W=32`, `atoms_per_window_keys_H=128`, `max_num_atoms_per_token=23`.
- **Graph signatures**: input names, shapes, dtypes for trunk, diffusion_step, confidence. **All graphs declare boundary I/O as fp32**; for the fp16 graphs you must cast inputs Float32Array → Uint16Array (fp16-packed) before `session.run` and cast outputs back. ORT-Web's `Tensor("float16", Uint16Array, shape)` accepts this directly. The int8 graphs keep fp32 boundaries.
- **`feats_spec`**: every one of the 78 feature tensors the trunk and diffusion-step graphs consume (`token_pad_mask`, `atom_pad_mask`, `ref_pos`, `ref_element`, …), with concrete shape and dtype. The biocircus side must reproduce the Boltz feature pipeline that builds this dict from a raw protein sequence; the `feats_spec` is your target schema.
- **Orchestration pseudocode** for both the recycling loop and the diffusion sampling loop, copy-pastable into a TS port.
- **Cα-extraction recipe** (see WARNING below).
- **pLDDT decoding recipe**: softmax over 50 bins, then expected value × 100.

---

## Three things you MUST not get wrong

These trip the biocircus integrator hard if missed. Each is documented in `reports/export_plan.md` (full pitfall catalogue) with a postmortem.

### 1. Use `token_to_center_atom`, NEVER `token_to_rep_atom`, for Cα extraction

```
ca_coords = einsum('bna,bad->bnd', feats['token_to_center_atom'], atom_coords)
```

- `token_to_center_atom` → Cα (`res_to_center_atom` in `boltz/data/const.py`)
- `token_to_rep_atom` → **Cβ** for non-Gly residues, Cα only for Gly (`res_to_disto_atom`)

`token_to_rep_atom` is the distogram input and is correctly used *inside* the graphs. It is **not** the right tensor for the final ribbon-coord extraction. Using it produces a visibly tangled rope with 5.3 Å Cα-Cα distances. This was Phase 4's worst bug; the postmortem is **P-9** in `export_plan.md`. The Cα-Cα consecutive distance is the canonical sanity-check — should be 3.78 ± 0.04 Å.

### 2. `sigma` is a `[B]` tensor, not a scalar

The diffusion-step graph's `sigma` input is a **1-dimensional tensor of length B**, not a Python float. ORT will error if you pass a 0-D tensor or scalar. In the sampling loop, set `sigma = [t_hat]` where `t_hat = sigma_tm * (1 + gamma)` (computed in TS), wrapped as a Float32Array of shape `[1]`.

### 3. Drive the recycling and sampling loops in JS, NEVER unroll into the graph

Both graphs export **one** iteration only:
- Trunk = one recycling pass. JS calls it `recycling_steps + 1` times, feeding the previous `(s, z)` back as `(s_prev, z_prev)`.
- Diffusion-step = one denoising step. JS calls it `sampling_steps` times with the Karras schedule.

This is the entire reason the architecture is two graphs not one. It also lets biocircus show step-progress in the UI and offer "fast/balanced/quality" presets via the step count.

---

## Orchestration loop reference

The Python reference implementation is `scripts/boltz_orchestrate.py` in this repo. The biocircus TS port needs the same structure. Pseudocode (with all the gotchas baked in):

```ts
// 1. Build feats dict (must match meta.json#feats_spec exactly)
const feats = buildFeats(sequence);   // 78 tensors

// 2. Recycling loop (loop length = recycling_steps + 1; meta.default = 1 → 2 calls)
let s_prev = zeros([B, N, token_s]);
let z_prev = zeros([B, N, N, token_z]);
let s, z, q, c, aeb, adb, ttb, s_inputs;
for (let i = 0; i <= recycling_steps; i++) {
    ({ s, z, pdistogram, q, c, atom_enc_bias: aeb, atom_dec_bias: adb,
       token_trans_bias: ttb, s_inputs } = await trunk.run({ ...feats, s_prev, z_prev }));
    s_prev = s; z_prev = z;
}

// 3. Build sigma schedule from meta.diffusion
const sigmas = buildKarrasSchedule(steps, sigma_min, sigma_max, sigma_data, rho);
const gammas = sigmas.map(σ => σ > gamma_min ? gamma_0 : 0);

// 4. Sampling loop
let atom_coords = scale(randn([B, A, 3]), sigmas[0]);
let atom_coords_denoised = null;
for (let step = 0; step < steps; step++) {
    const σtm = sigmas[step], σt = sigmas[step + 1], γ = gammas[step + 1];
    const [R, tr] = haarRotationPlusTranslation(B);  // see weighted_rigid_align note below
    atom_coords = applyAffine(meanCenter(atom_coords), R, tr);
    if (atom_coords_denoised) atom_coords_denoised = applyAffine(meanCenter(atom_coords_denoised), R, tr);
    const tHat = σtm * (1 + γ);
    const noiseVar = noise_scale ** 2 * (tHat ** 2 - σtm ** 2);
    const eps = scale(randn([B, A, 3]), Math.sqrt(Math.max(noiseVar, 0)));
    const atom_coords_noisy = add(atom_coords, eps);
    atom_coords_denoised = await diffusionStep.run({
        ...feats, s, s_inputs, q, c, atom_enc_bias: aeb, atom_dec_bias: adb,
        token_trans_bias: ttb, x_noisy: atom_coords_noisy, sigma: [tHat],
    }).x_denoised;
    if (alignment_reverse_diff) {
        const aligned = kabschWeightedAlign(atom_coords_noisy, atom_coords_denoised,
                                             feats.atom_pad_mask, feats.atom_pad_mask);
        atom_coords_noisy = aligned;  // cast back to fp16 if running fp16 graph
    }
    const denoisedOverSigma = scale(sub(atom_coords_noisy, atom_coords_denoised), 1 / tHat);
    atom_coords = add(atom_coords_noisy,
                      scale(denoisedOverSigma, step_scale * (σt - tHat)));
}

// 5. Confidence pass
const { plddt_logits, pae_logits } = await confidence.run({
    ...feats, s_inputs, s, z, x_pred: atom_coords,
});

// 6. Cα extraction (USE token_to_center_atom!)
const ca = einsum("bna,bad->bnd", feats.token_to_center_atom, atom_coords);

// 7. pLDDT decode
const probs = softmax(plddt_logits, /*axis*/ -1);
const binCenters = range(50).map(i => (i + 0.5) / 50);   // 0.01, 0.03, ..., 0.99
const plddt = sumProduct(probs, binCenters, /*axis*/ -1);  // [B, N], range [0, 1]
const plddtForDisplay = scale(plddt, 100);

// 8. Render in Mol*: build a PDB or mmCIF from ca (one record per residue, atom "CA"),
//    colour by plddtForDisplay using the standard AF2 palette.
```

### TS helpers you'll need

- **Haar-uniform random rotation matrices**. The Python uses `pytorch3d.transforms.random_rotations(...)`. Implementation: sample a quaternion from the 4-sphere (4 i.i.d. normals, normalise), convert to rotation matrix. ~15 lines of TS.
- **Kabsch / weighted_rigid_align**. The Python implementation in `boltz/model/loss/diffusionv2.py` builds the weighted covariance, SVD, det fix, and rotation. ~30 lines of TS. **TF.js or `numjs` does the SVD if you don't want to ship a separate linalg dep**, but the `[B, A, 3]` matrix is small (A=352, so the cov is 3×3) — eigendecomposition of a 3×3 matrix has a closed form. Worth a small utility.
- **Karras sigma schedule**. The formula is in `meta.json` and is a few lines.
- **fp16 packing** for fp16-tier inference. Use `Float16Array` if available in the browser, else pack manually into `Uint16Array` (IEEE 754 binary16). ORT-Web's `Tensor("float16", ...)` accepts the `Uint16Array` view directly.

---

## Validation gate for the integration

When wiring up, validate at each level:

1. **Single trunk pass numerically matches the Python orchestrator.** Run the Python `boltz_orchestrate.py` with `--out tmp.pdb`, then run your TS orchestrator with the same `feats` dict (you'll need to dump it once from Python). Max abs diff on `s`, `z`, `pdistogram` should be < 2e-3 for fp32, < 5e-3 for fp16.
2. **Final Cα geometry sanity-check.** Cα-Cα consecutive distance must be 3.78 ± 0.04 Å. If it's anywhere near 5.3 Å, you've hit pitfall P-9 (wrong atom map).
3. **Visual fold.** Render in Mol* and confirm it looks protein-like (compact globule with helix/sheet hints, not a tangled rope or extended chain). For 1CRN specifically: small ovoid ~25 Å diameter.
4. **Inter-seed noise floor.** Run the prediction 3 times with different random seeds. Pairwise Kabsch-aligned Cα RMSD should be 5–8 Å (single-seq no-MSA is intrinsically high-variance because the model has no co-evolution signal; pLDDT will be ~0.45). With MSA enabled in a v0.1, that drops to <2 Å.

If any of those fail, the bug is in the TS orchestration (or the feats pipeline), not in the ONNX graphs. The graphs have been verified at `max_abs_diff < 3e-4` against PyTorch per-step.

---

## Known limitations

- **Concrete shapes only** (N=46, A=352). The same ONNX graphs will fail to load with a different L. Phase 5+ work: re-export with `dynamic_shapes` declared on N and A axes. Expect a fresh round of pitfall hits (data-dependent guards on padding ops).
- **int8 has slight bond-length compression** (3.45 ± 0.18 Å vs 3.78 ± 0.014 Å for fp32/fp16). Globally the fold is correct; up close the chain looks more "fuzzy." Acceptable for ribbon rendering; mention it in the biocircus UI if you offer a precision-tier picker ("low-memory mode may show slight bond variance").
- **No `pair_chains_iptm` scalar.** The Boltz Python CLI computes it on the way out via `compute_ptms`. The confidence ONNX graph exposes the `pae_logits` it would consume; if biocircus wants pTM/ipTM, port the `compute_ptms` formula from `boltz/model/layers/confidence_utils.py` to TS. ~50 lines.
- **Single-protein-chain only.** Multi-chain inputs (`asym_id` varying) should still work mathematically, but only single-chain (`asym_id == 0`) has been validated. The `use_separate_heads=True` confidence path is wired correctly for multi-chain inputs; the chain-mask branches are exported.
- **Inputs are 78 tensors.** The biocircus feature pipeline must produce all 78 with exact shapes and dtypes (see `meta.json#feats_spec`). Missing or mis-shaped tensors will fault the ORT session at `run()` time.

---

## Repo structure in `boltz-dev` (for reference, if a question forces re-export)

- `boltz/` — the upstream Boltz repo, clone of `jwohlwend/boltz` at HEAD. **Do not edit** — all conversion patches live in our wrapper scripts.
- `phase0_reference/1CRN.yaml`, `1CRN_baseline.pdb` — input + golden PyTorch output.
- `phase1_trace/audit_boltz2.md` — initial ONNX-export blocker audit.
- `phase2_trunk/trunk.onnx (+ .data)` — fp32 trunk graph.
- `phase3_diffusion/diffusion_step.onnx (+ .data)` — fp32 diffusion-step graph.
- `phase4/` — validation report + ORT orchestration outputs.
- `phase5_quant/` — fp16 + int8 graphs (trunk, diffusion_step, confidence), validation report.
- `phase5b_confidence/` — fp32 confidence graph + meta.json.
- `scripts/` — export, quantize, orchestrate, build_meta, validate utilities. **The single-source-of-truth Python orchestrator is `boltz_orchestrate.py`.**
- `hf_upload/push.sh`, `push_phase5b.sh` — HF upload scripts.
- `EXPORT_PLAN.md` — phased plan + pitfall catalogue (P-1 through P-9).

If something forces a re-export (e.g., dynamic shapes, MSA support), the `scripts/` directory is set up to re-run end-to-end: `export_trunk.py` → `export_diffusion_step.py` → `export_confidence.py` → `quantize.py` → `boltz_orchestrate.py` for validation. Each script is self-contained and uses the same CLI-hook capture to instantiate the live Boltz-2 model with its checkpoint weights.

---

Generated 2026-05-15. Owner: boltz-dev. Contact: see biocircus repo CODEOWNERS.
