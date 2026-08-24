# H3 Turbo LoRA Bridge

A compact ComfyUI bridge for MiniMax H3 full-AdaLN LoRAs on pruned/curve H3 models.

## v0.6.26.1 — bridge rename

This build keeps the proven v0.6.26 integrated logic, with the node/repository renamed to **H3 Turbo LoRA Bridge**. It folds the **H3 AdaLN LoRA Fix** into the same node. A separate PlagueKind/LinhTinh AdaLN-fix node is no longer required.

The node first applies the 208 non-AdaLN backbone modules with the existing H3-Turbo routing, attaches the 51 AdaLN adapters through ComfyUI's native LoRA parser, then optionally rebases those AdaLN patches onto the pruned H3 curve basis using the vendored PlagueKind port core.

## Controls

- `model`
- `lora_name`
- `adaln_strength` — direct/absolute AdaLN LoRA strength, default `1.0`
- `backbone_strength` — direct/absolute backbone LoRA strength, default `0.45`
- `adaln_fix_mode`
  - `port` — default; rebase incompatible dense AdaLN LoRA patches onto the model's curve basis
  - `strip` — remove incompatible AdaLN patches
  - `off` — leave the raw mismatched patches untouched
- `console_log` — verbose loader diagnostics

There is **no `strength_model` master multiplier** in this build. The two strengths are independent:

```text
adaln_strength = 1.00     -> AdaLN effective strength = 1.00
backbone_strength = 0.45  -> backbone effective strength = 0.45
```

Set both strengths to `0` if you want this bridge to contribute no LoRA effect.

## Recommended graph

```text
UNETLoader
  -> H3 Turbo LoRA Bridge
  -> H3 Memory Optimization (optional)
  -> H3 Sparse Attention (optional)
  -> MiniMaxH3SigmaShift
  -> sampler
```

Do **not** apply the same LoRA again with `LoraLoaderModelOnly` or another LoRA loader.

A standalone `H3 AdaLN LoRA Fix` node after this bridge is redundant. If one is accidentally left in the graph it should simply find no mismatched AdaLN patches after `port`, but removing it keeps the workflow clear.

## AdaLN basis / grid

The bundled AdaLN port logic is the PlagueKind implementation. For a pruned H3 model it needs a trustworthy mapping between the dense 2688-D time-embedding curve and the model's 8-D AdaLN table.

It can derive/cache a basis from compatible local H3 checkpoints, or use the pre-baked grid file:

```text
ComfyUI/models/h3_adaln/h3_silu_temb_grid.safetensors
```

With the tested grid/model combination, a successful port typically logs a line such as:

```text
[H3AdaLN] AdaLN LoRA fix: ported 51 patch(es) across 51 key(s) dense->curve, ... (residual 1.68e-03)
```

The PlagueKind core rejects a basis fit above its own safety threshold and degrades to `strip` rather than applying an untrusted conversion.

## Backbone behavior

The backbone path is inherited from v0.6.25 / v0.6.21:

- ordinary compatible backbone modules use the H3-Turbo bypass path
- supported INT8-fused FC2 modules use the native merged/weight-patch path
- no backbone rank compression
- no custom runtime AdaLN wrappers

Typical ConvRot H3:

```text
Backbone patches validated: 158 bypass ... 50 INT8-fused/merged ... total=208/208
```

Some mixed checkpoints use all-bypass routing instead:

```text
Backbone patches validated: 208 bypass ... 0 INT8-fused/merged ... total=208/208
```

The first-forward verifier remains enabled and should confirm that the bypass path is actually executing.

## H3-Optimizations compatibility

When Zironic/H3-Optimizations is installed, the v0.6.21 FC1 compatibility bridge remains available for its ConvRot two-slice MLP path. The H3-Optimizations source tree is not modified.

On a compatible ConvRot model you may see:

```text
[H3 Optimizations] patched 50 MLP blocks: mode=mlp_chunked_convrot_2slice ...
[H3 Optimizations] ... mlp=convrot_int8_two_slice ...
[H3TurboLoRA][INFO] H3 Memory exact-FC1 bridge first-forward OK ...
```

If H3-Optimizations reports `mlp=preserve_upstream_mlp`, its MLP provider is not active for that checkpoint format; this bridge does not force or monkey-patch H3-Optimizations into accepting an unsupported MLP layout.

The bridge was reviewed against Zironic/H3-Optimizations commit `1c79d5ffa616261dcf93a7fa930611cda64a33ef`. Future upstream contract changes may require compatibility review.

## Dependencies

Required:

- ComfyUI with MiniMax H3 support
- `ComfyUI-MiniMax-H3-Turbo` under `ComfyUI/custom_nodes`
- the target MiniMax H3 model and LoRA files

Optional:

- Zironic/H3-Optimizations for Memory/Sparse optimizations

A separate installation of `ComfyUI-PlagueKind-Nodes` is **not required** for the integrated AdaLN fix. Only the relevant AdaLN port core is vendored here.

## Installation

Copy or clone this repository into:

```text
ComfyUI/custom_nodes/ComfyUI-H3-Turbo-LoRA-Bridge
```

Restart ComfyUI after replacing the node source.

## Attribution and licenses

This project contains/adapts code from two upstream projects:

1. [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) — Apache License 2.0. The repository root `LICENSE` contains the Apache-2.0 text used for this project.
2. [PlagueKind/ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes), specifically the H3 AdaLN LoRA Fix core — MIT License. The vendored source and its MIT license are in `plaguekind_h3_adaln/`.

This repository does **not** bundle MiniMax model weights, Turbo LoRA weights, or the `h3_silu_temb_grid.safetensors` data file.
