# Portions of this file are adapted from ComfyUI-MiniMax-H3-Turbo
# by Larryvrh, licensed under the Apache License 2.0.
# Upstream: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
# This file has been modified for full-width AdaLN LoRA support on pruned/curve MiniMax-H3 models.
# See LICENSE and README.md for attribution and licensing details.

"""
H3 Turbo LoRA Bridge

MiniMax-H3 LoRA bridge for full-width AdaLN LoRAs used with pruned/curve H3
checkpoints.

v0.6.26.1 keeps the v0.6.26 integrated path and the PlagueKind H3 AdaLN
LoRA port into one node:
  * the 208 non-AdaLN backbone modules keep the proven H3-Turbo custom routing
    (activation-space bypass plus native FC2 merge);
  * the 51 AdaLN adapters are first attached as ordinary ComfyUI LoRA patches;
  * the bundled PlagueKind port core can then rebase those dense AdaLN patches
    onto the pruned model's 8-D curve basis inside this same loader.

No separate H3 AdaLN LoRA Fix node is required.  The port/strip/off behavior is
exposed as ``adaln_fix_mode``.  The vendored PlagueKind core remains MIT licensed
and is kept in ``plaguekind_h3_adaln/``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

import folder_paths

from .plaguekind_h3_adaln import adaln_patch as _adaln_patch

VERSION = "0.6.26.1"
PREFIX = "[H3TurboLoRA]"

_BACKEND_MODULE = None
_LOG_ENABLED = False


def _info(msg: str):
    if _LOG_ENABLED:
        print(f"{PREFIX}[INFO] {msg}", flush=True)


def _warn(msg: str):
    print(f"{PREFIX}[WARN] {msg}", flush=True)


def _error(msg: str):
    print(f"{PREFIX}[ERROR] {msg}", flush=True)


# H3-Optimizations compatibility -------------------------------------------------
#
# The normal H3 Turbo loader deliberately keeps ordinary quantized Linear LoRAs
# in activation-space bypass mode.  That avoids folding a small BF16 LoRA delta
# into an INT8/FP8 base weight where requantization can soften it.
#
# Zironic/H3-Optimizations' Memory Optimization has one special H3 ConvRot MLP
# path which is faster and more memory efficient precisely because it does not
# call fc1.forward(): it acquires the quantized fc1 weight and evaluates two
# feature tiles directly.  A normal BypassForwardHook therefore cannot run for
# fc1 in that path.  fc2 does not have this problem in this loader: ConvRot INT8
# fc2 has always used the backend's native Comfy weight-patch path because stock
# H3's fused SwiGLU+fc2 path also bypasses fc2.forward().
#
# v0.6.21 keeps the proven v6.14 routing unchanged and installs a tiny runtime
# bridge only at H3 Memory's ConvRot two-slice fc1 boundary.  The base fc1 still
# runs through H3 Memory's exact INT8 ConvRot kernel; immediately afterwards we
# add the same low-rank BF16 residual that _FrugalLoRA.bypass_forward() would
# have added.  The residual is produced per H3 feature tile, so the bridge never
# materializes a full-width fc1 activation and does not disable the two-slice
# memory optimization.
#
# Nothing is patched in the H3-Optimizations source tree.  We replace only the
# class reference used by its memory.forward module with a guarded subclass.  On
# models without this loader's _FrugalLoRA fc1 hook the subclass is a no-op.
_MEMORY_BRIDGE_INSTALLED = False
_MEMORY_BRIDGE_FIRST_FORWARD = False
_MEMORY_BRIDGE_CLASS = None


def _resolve_h3_memory_forward_module():
    """Return the *live* H3-Optimizations memory.forward module.

    ComfyUI can load custom-node packages under generated package names, so the
    canonical ``h3_optimizations.memory.forward`` import is not guaranteed to be
    available when this node executes.  Prefer an already-loaded module by file
    identity, then try canonical/dynamic package imports, and finally add only the
    discovered H3-Optimizations plugin root to sys.path long enough to import its
    canonical package.  We must patch the live module instance because its
    ``_open_mlp`` resolves ``ConvRotTwoSliceMLP`` from that module's globals.
    """

    suffix = str(Path("h3_optimizations") / "memory" / "forward.py").replace("\\", "/")

    # 1) Best case: H3-Optimizations has already imported memory.forward.  This
    # catches canonical as well as ComfyUI-generated package names.
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            normalized = str(filename).replace("\\", "/")
            if normalized.endswith(suffix):
                return module
        if name == "h3_optimizations.memory.forward":
            return module

    # 2) Canonical package import, used by current upstream when its root alias is
    # already registered.
    try:
        return importlib.import_module("h3_optimizations.memory.forward")
    except ModuleNotFoundError:
        pass

    # 3) A custom-node package may be present in sys.modules under a generated
    # parent name.  Try importing memory.forward relative to each live package
    # whose source directory is h3_optimizations.
    for name, module in list(sys.modules.items()):
        if module is None or not getattr(module, "__path__", None):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            package_dir = Path(filename).resolve().parent
        except Exception:
            continue
        if package_dir.name != "h3_optimizations":
            continue
        try:
            return importlib.import_module(name + ".memory.forward")
        except ModuleNotFoundError:
            continue

    # 4) Last resort: locate the installed plugin source and make its package root
    # importable.  This does not modify H3-Optimizations and mirrors what its own
    # entry point does when registering the canonical h3_optimizations alias.
    custom_nodes = Path(folder_paths.base_path) / "custom_nodes"
    candidates = []
    try:
        for child in custom_nodes.iterdir():
            if not child.is_dir():
                continue
            forward_py = child / "h3_optimizations" / "memory" / "forward.py"
            if forward_py.is_file():
                candidates.append(child)
    except Exception:
        candidates = []

    for root in candidates:
        root_s = str(root)
        inserted = root_s not in sys.path
        if inserted:
            sys.path.insert(0, root_s)
        try:
            module = importlib.import_module("h3_optimizations.memory.forward")
            # Keep the root on sys.path after a successful import.  H3-Optimizations
            # lazily imports additional sibling providers at runtime.
            inserted = False
            return module
        except ModuleNotFoundError:
            pass
        finally:
            if inserted:
                try:
                    sys.path.remove(root_s)
                except ValueError:
                    pass

    return None


def _install_h3_memory_fc1_bridge(backend):
    """Install the exact fc1-LoRA bridge for H3 Memory Optimization if present.

    Returns True when the compatible bridge is installed/already installed and
    False when H3-Optimizations is not installed.  Structural incompatibilities
    are warned about rather than silently changing the loader's normal behavior.
    """
    global _MEMORY_BRIDGE_INSTALLED, _MEMORY_BRIDGE_CLASS

    if _MEMORY_BRIDGE_INSTALLED:
        try:
            current_forward = _resolve_h3_memory_forward_module()
            current_cls = (
                getattr(current_forward, "ConvRotTwoSliceMLP", None)
                if current_forward is not None else None
            )
            if getattr(current_cls, "_h3turbo_exact_fc1_bridge", False):
                _MEMORY_BRIDGE_CLASS = current_cls
                return True
        except ModuleNotFoundError:
            return False
        except Exception:
            pass
        # The optimization module was reloaded or replaced after bridge install.
        # Re-evaluate its current class instead of silently trusting a stale flag.
        _MEMORY_BRIDGE_INSTALLED = False
        _MEMORY_BRIDGE_CLASS = None

    frugal_cls = getattr(backend, "_FrugalLoRA", None)
    if frugal_cls is None:
        _warn(
            "H3 Memory compatibility bridge unavailable: the installed H3 Turbo "
            "backend exposes no _FrugalLoRA class. Normal loader behavior is unchanged."
        )
        return False

    try:
        memory_forward = _resolve_h3_memory_forward_module()
    except Exception as exc:
        _warn(
            "H3 Memory compatibility bridge could not resolve the live "
            f"H3-Optimizations memory.forward module: {type(exc).__name__}: {exc}. "
            "Normal loader behavior is unchanged."
        )
        return False
    if memory_forward is None:
        # H3-Optimizations is optional. Keep normal loader behavior unchanged.
        _info(
            "H3 Memory compatibility bridge not installed: H3-Optimizations "
            "memory.forward is not currently available."
        )
        return False

    base_cls = getattr(memory_forward, "ConvRotTwoSliceMLP", None)
    if base_cls is None:
        _warn(
            "H3 Memory compatibility bridge unavailable: ConvRotTwoSliceMLP was "
            "not found. Normal loader behavior is unchanged."
        )
        return False

    if getattr(base_cls, "_h3turbo_exact_fc1_bridge", False):
        _MEMORY_BRIDGE_CLASS = base_cls
        _MEMORY_BRIDGE_INSTALLED = True
        return True

    required = ("__enter__", "release", "fc1_fc2")
    missing = [name for name in required if not hasattr(base_cls, name)]
    if missing:
        _warn(
            "H3 Memory compatibility bridge refused an unknown "
            f"ConvRotTwoSliceMLP API (missing {missing}). Normal loader behavior "
            "is unchanged; do not combine this loader with H3 Memory Optimization "
            "until the bridge is updated."
        )
        return False

    class H3TurboLoRAConvRotTwoSliceMLP(base_cls):
        """H3 Memory two-slice MLP plus exact activation-space fc1 LoRA."""

        _h3turbo_exact_fc1_bridge = True
        _h3turbo_bridge_base = base_cls

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._h3turbo_fc1_adapter = None
            self._h3turbo_base_convrot_linear = None
            self._h3turbo_down_x_id = None
            self._h3turbo_down = None
            # Per-held-session device cache.  H3 Memory creates one held MLP
            # session per block/denoise step and then evaluates many token slabs.
            # Keeping A plus the two B feature tiles here avoids re-copying the
            # full FC1 LoRA from CPU (and re-splitting B) for every slab/tile,
            # while release() still gives the memory back at block end.
            self._h3turbo_fc1_weight_cache = None

            try:
                fc1 = self.mlp.fc1
                owner = getattr(getattr(fc1, "forward", None), "__self__", None)
                adapter = getattr(owner, "adapter", None)
                if isinstance(adapter, frugal_cls):
                    weights = getattr(adapter, "weights", None)
                    if isinstance(weights, (tuple, list)) and len(weights) >= 2:
                        up, down = weights[0], weights[1]
                        if (
                            torch.is_tensor(up)
                            and torch.is_tensor(down)
                            and up.ndim == 2
                            and down.ndim == 2
                            and int(up.shape[1]) == int(down.shape[0])
                            and int(up.shape[0]) % 4 == 0
                        ):
                            self._h3turbo_fc1_adapter = adapter
                            self._h3turbo_base_convrot_linear = self.convrot_linear
                            self.convrot_linear = self._h3turbo_convrot_linear
            except Exception:
                # Detection must never alter ordinary H3 Memory execution.  If a
                # future Comfy bypass object has a different shape, remain a no-op.
                self._h3turbo_fc1_adapter = None
                self._h3turbo_base_convrot_linear = None

        @staticmethod
        def _h3turbo_scalar(value):
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError("LoRA alpha must be scalar")
                return float(value.detach().float().cpu().item())
            return float(value)

        def _h3turbo_lora_parts(self, x):
            adapter = self._h3turbo_fc1_adapter
            weights = adapter.weights
            up_src, down_src = weights[0], weights[1]
            alpha = weights[2] if len(weights) > 2 else None

            rank = int(down_src.shape[0])
            if rank <= 0:
                raise ValueError("LoRA rank must be positive")

            # Exact same BF16 tensors/math as _FrugalLoRA.bypass_forward(), but
            # materialize A/B on the active device only ONCE for the complete
            # H3 Memory held session.  v6.18 did .to() and rebuilt both B tiles
            # inside every ConvRot tile call, which could turn a 4096-row memory
            # chunk into repeated PCIe copies and many avoidable cat kernels.
            key = (
                str(x.device),
                str(x.dtype),
                tuple(int(v) for v in up_src.shape),
                tuple(int(v) for v in down_src.shape),
                int(up_src.data_ptr()),
                int(down_src.data_ptr()),
            )
            cache = self._h3turbo_fc1_weight_cache
            if not isinstance(cache, dict) or cache.get("key") != key:
                down = down_src.to(device=x.device, dtype=x.dtype)
                up = up_src.to(device=x.device, dtype=x.dtype)

                half = int(up.shape[0]) // 2
                tile_width = half // 2
                up_tile0 = torch.cat(
                    (up[:tile_width], up[half : half + tile_width]), dim=0
                ).contiguous()
                up_tile1 = torch.cat(
                    (up[tile_width:half], up[half + tile_width :]), dim=0
                ).contiguous()
                cache = {
                    "key": key,
                    "down": down,
                    "up_tiles": (up_tile0, up_tile1),
                }
                self._h3turbo_fc1_weight_cache = cache

            base_scale = 1.0 if alpha is None else self._h3turbo_scalar(alpha) / rank
            multiplier = float(getattr(adapter, "multiplier", 1.0))
            return cache["up_tiles"], cache["down"], base_scale * multiplier

        def _h3turbo_tile_index(self, qdata):
            tiles = self.tiles
            if tiles is None:
                return None
            for index, tile in enumerate(tiles):
                candidate = tile.get("fc1_weight")
                if qdata is candidate:
                    return index
                try:
                    if (
                        torch.is_tensor(qdata)
                        and torch.is_tensor(candidate)
                        and tuple(qdata.shape) == tuple(candidate.shape)
                        and qdata.data_ptr() == candidate.data_ptr()
                    ):
                        return index
                except Exception:
                    pass
            return None

        def _h3turbo_convrot_linear(self, x, qdata, scale, input_act=None):
            global _MEMORY_BRIDGE_FIRST_FORWARD

            out = self._h3turbo_base_convrot_linear(
                x, qdata, scale, input_act=input_act
            )
            if input_act is not None or self._h3turbo_fc1_adapter is None:
                return out

            tile_index = self._h3turbo_tile_index(qdata)
            if tile_index is None:
                # Unknown fc1 tile geometry: fail loudly rather than silently
                # dropping the LoRA and producing a deceptively valid blurry run.
                raise RuntimeError(
                    "H3 Turbo LoRA Memory bridge could not identify the fc1 "
                    "two-slice tile; H3-Optimizations internals may have changed."
                )

            up_tiles, down, lora_scale = self._h3turbo_lora_parts(x)

            x_id = id(x)
            if self._h3turbo_down_x_id != x_id or self._h3turbo_down is None:
                self._h3turbo_down = F.linear(x, down)
                self._h3turbo_down_x_id = x_id

            up_tile = up_tiles[int(tile_index)]
            if int(down.shape[1]) != int(x.shape[-1]) or int(up_tile.shape[0]) != int(out.shape[-1]):
                raise RuntimeError(
                    "H3 Turbo LoRA Memory bridge found incompatible fc1 LoRA/tile "
                    f"geometry: x={tuple(x.shape)}, A={tuple(down.shape)}, "
                    f"B_tile={tuple(up_tile.shape)}, base_out={tuple(out.shape)}"
                )
            # Preserve the activation-space LoRA correction at the exact H3
            # fc1 tile boundary, but accumulate the low-rank GEMM directly into
            # the resident ConvRot output.  This removes v6.19's full-size BF16
            # delta allocation and the separate add kernel while staying on the
            # current CUDA stream (no allocator pressure from side streams).
            #
            # Mathematically this is identical to:
            #   out += F.linear(down_act, up_tile) * lora_scale
            # but cuBLAS can use GEMM beta/alpha epilogue semantics and write
            # directly into ``out``.  If the tensor layout/API does not support
            # in-place addmm, fall back to the v6.19 F.linear + add_ path.
            used_fused_accumulate = False
            try:
                if out.is_contiguous() and self._h3turbo_down.is_contiguous():
                    out_2d = out.view(-1, out.shape[-1])
                    down_2d = self._h3turbo_down.view(-1, self._h3turbo_down.shape[-1])
                    if int(out_2d.shape[0]) == int(down_2d.shape[0]):
                        out_2d.addmm_(
                            down_2d,
                            up_tile.transpose(0, 1),
                            beta=1.0,
                            alpha=float(lora_scale),
                        )
                        used_fused_accumulate = True
            except (RuntimeError, TypeError):
                used_fused_accumulate = False

            if not used_fused_accumulate:
                out.add_(F.linear(self._h3turbo_down, up_tile), alpha=lora_scale)

            if tile_index == 1:
                self._h3turbo_down = None
                self._h3turbo_down_x_id = None

            if not _MEMORY_BRIDGE_FIRST_FORWARD:
                _MEMORY_BRIDGE_FIRST_FORWARD = True
                _info(
                    "H3 Memory exact-FC1 bridge first-forward OK: "
                    f"rank={int(down.shape[0])}, dtype={x.dtype}, device={x.device}; "
                    "base ConvRot two-slice path preserved; LoRA residual accumulated "
                    "before SwiGLU; schedule=same-stream in-place addmm "
                    f"({'active' if used_fused_accumulate else 'fallback F.linear+add_'}); "
                    "per-session A/B cache active."
                )
            return out

        def release(self):
            self._h3turbo_down = None
            self._h3turbo_down_x_id = None
            self._h3turbo_fc1_weight_cache = None
            return super().release()

    memory_forward.ConvRotTwoSliceMLP = H3TurboLoRAConvRotTwoSliceMLP
    _MEMORY_BRIDGE_CLASS = H3TurboLoRAConvRotTwoSliceMLP
    _MEMORY_BRIDGE_INSTALLED = True
    _info(
        "H3 Memory compatibility bridge installed: v6.14 backbone routing is "
        "unchanged; ConvRot two-slice fc1 receives the exact bypass-LoRA residual "
        "in activation space."
    )
    return True


def _find_backend_init() -> Path:
    custom_nodes = Path(folder_paths.base_path) / "custom_nodes"

    preferred = (
        custom_nodes / "ComfyUI-MiniMax-H3-Turbo" / "__init__.py",
        custom_nodes / "ComfyUI_MiniMax_H3_Turbo" / "__init__.py",
    )
    for p in preferred:
        if p.is_file():
            return p.resolve()

    # Fallback: identify the H3 Turbo backend by its class and bundled E-grid.
    if custom_nodes.is_dir():
        for child in custom_nodes.iterdir():
            if not child.is_dir():
                continue
            init_file = child / "__init__.py"
            grid_file = child / "h3_silu_temb_grid.safetensors"
            if not (init_file.is_file() and grid_file.is_file()):
                continue
            try:
                txt = init_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "MiniMaxH3TurboLoRA" in txt and "silu_t_emb_grid" in txt:
                return init_file.resolve()

    raise RuntimeError(
        "H3 Turbo backend not found. Install/update "
        "ComfyUI-MiniMax-H3-Turbo under ComfyUI/custom_nodes, "
        "restart ComfyUI, then retry."
    )


def _already_loaded_module_for(init_file: Path):
    wanted = os.path.normcase(str(init_file.resolve()))
    for mod in tuple(sys.modules.values()):
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        try:
            current = os.path.normcase(str(Path(mod_file).resolve()))
        except Exception:
            continue
        if current == wanted and hasattr(mod, "MiniMaxH3TurboLoRA"):
            return mod
    return None


def _load_backend_module():
    global _BACKEND_MODULE
    if _BACKEND_MODULE is not None:
        return _BACKEND_MODULE

    init_file = _find_backend_init()

    loaded = _already_loaded_module_for(init_file)
    if loaded is not None:
        _info(f"Reusing already-loaded H3 Turbo backend: {init_file}")
        _BACKEND_MODULE = loaded
        return loaded

    module_name = "_h3_turbo_backend_runtime"
    if module_name in sys.modules:
        mod = sys.modules[module_name]
        if hasattr(mod, "MiniMaxH3TurboLoRA"):
            _BACKEND_MODULE = mod
            return mod

    _info(f"Loading H3 Turbo backend from: {init_file}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(init_file),
        submodule_search_locations=[str(init_file.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for H3 Turbo backend: {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if not hasattr(module, "MiniMaxH3TurboLoRA"):
        raise RuntimeError(
            f"H3 Turbo backend loaded from {init_file}, but class "
            "MiniMaxH3TurboLoRA was not found."
        )

    _BACKEND_MODULE = module
    return module


def _shape_tuple(shape):
    return tuple(int(x) for x in shape)


def _inspect_lora_header(path: str):
    """
    Inspect only safetensors metadata/header. Tensor payloads are not loaded.
    Returns a compact report used to reject obvious incompatible AdaLN layouts.
    """
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(
            "safetensors.safe_open is unavailable; cannot perform the preflight "
            f"shape check. Original error: {exc}"
        ) from exc

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        shapes = {k: _shape_tuple(f.get_slice(k).get_shape()) for k in keys}

    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"

    a_modules = {k[:-len(a_suffix)] for k in keys if k.endswith(a_suffix)}
    b_modules = {k[:-len(b_suffix)] for k in keys if k.endswith(b_suffix)}

    paired = sorted(a_modules & b_modules)
    missing_b = sorted(a_modules - b_modules)
    missing_a = sorted(b_modules - a_modules)
    adaln = [m for m in paired if "adaln_proj" in m]
    backbone = [m for m in paired if "adaln_proj" not in m]

    return {
        "keys": keys,
        "metadata": metadata,
        "shapes": shapes,
        "paired": paired,
        "missing_a": missing_a,
        "missing_b": missing_b,
        "adaln": adaln,
        "backbone": backbone,
    }


def _validate_adaln_layout(report, pruned: bool):
    if report["missing_a"] or report["missing_b"]:
        samples = (report["missing_a"][:3] + report["missing_b"][:3])
        raise ValueError(
            "LoRA contains orphan A/B tensors. "
            f"missing_A={len(report['missing_a'])}, "
            f"missing_B={len(report['missing_b'])}, sample={samples}"
        )

    if not report["paired"]:
        raise ValueError(
            "No standard '.lora_A.weight' + '.lora_B.weight' pairs were found. "
            "This file is not in the expected H3 LoRA layout."
        )

    if not pruned:
        return

    # Official full H3: time_embed_dim=2688. The pruned/curve base exposes an
    # 8-wide AdaLN input, but full AdaLN LoRA A tensors must still consume 2688.
    bad = []
    for module in report["adaln"]:
        a = report["shapes"][module + ".lora_A.weight"]
        b = report["shapes"][module + ".lora_B.weight"]

        if len(a) != 2 or len(b) != 2:
            bad.append((module, a, b, "A/B must be rank-2"))
            continue

        rank_a, in_dim = a
        out_dim, rank_b = b
        if rank_a != rank_b:
            bad.append((module, a, b, "A/B rank mismatch"))
            continue

        expected_out = 10752 if "final_layer.adaln_proj" in module else 96768
        if in_dim != 2688:
            bad.append((module, a, b, f"A input dim {in_dim} != 2688"))
        elif out_dim != expected_out:
            bad.append((module, a, b, f"B output dim {out_dim} != {expected_out}"))

    if bad:
        first = bad[0]
        raise ValueError(
            "AdaLN LoRA layout does not match the official full-width H3 "
            "projection expected by the runtime injector. "
            f"bad_count={len(bad)}, first={first}"
        )


def _inspect_model(model):
    if not hasattr(model, "model"):
        raise TypeError(
            f"Expected a ComfyUI MODEL/ModelPatcher, got {type(model)!r} "
            "without a '.model' attribute."
        )

    inner = model.model
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        raise TypeError(
            f"MODEL inner object {type(inner)!r} has no '.diffusion_model'. "
            "This does not look like the expected ComfyUI MiniMax-H3 MODEL."
        )

    pruned = bool(getattr(dm, "use_adaln_curves", False))
    config = getattr(inner, "model_config", None)

    _info(
        "Model: "
        f"model_config={type(config).__name__ if config is not None else 'unknown'}, "
        f"diffusion_model={type(dm).__name__}, "
        f"use_adaln_curves={pruned}"
    )

    if hasattr(dm, "sigma_shift_video") or hasattr(dm, "sigma_shift_audio"):
        _info(
            "H3 shifts visible on diffusion model: "
            f"video={getattr(dm, 'sigma_shift_video', 'n/a')}, "
            f"audio={getattr(dm, 'sigma_shift_audio', 'n/a')}"
        )

    if pruned:
        table = getattr(dm, "adaln_t_table", None)
        if table is not None:
            _info(f"Pruned AdaLN table shape={tuple(table.shape)}")
            if len(table.shape) != 2 or int(table.shape[-1]) != 8:
                _warn(
                    "This pruned H3 uses a non-standard AdaLN curve width "
                    f"{tuple(table.shape)}; the current H3 runtime bridge was "
                    "designed around the official 8-wide curve representation."
                )

    return dm, pruned


def _validate_backend_grid(backend, pruned: bool):
    if not pruned:
        return

    backend_file = Path(getattr(backend, "__file__", "")).resolve()
    grid_path = backend_file.parent / "h3_silu_temb_grid.safetensors"
    if not grid_path.is_file():
        raise FileNotFoundError(
            "H3 Turbo backend is present, but its required "
            f"h3_silu_temb_grid.safetensors is missing: {grid_path}"
        )

    try:
        from safetensors import safe_open
        with safe_open(str(grid_path), framework="pt", device="cpu") as f:
            if "silu_t_emb_grid" not in f.keys():
                raise ValueError(
                    f"{grid_path} does not contain tensor 'silu_t_emb_grid'."
                )
            shape = _shape_tuple(f.get_slice("silu_t_emb_grid").get_shape())
    except Exception as exc:
        raise RuntimeError(f"Could not validate H3 E-grid: {exc}") from exc

    _info(f"H3 E-grid: {grid_path.name}, shape={shape}")
    if len(shape) != 2 or shape[1] != 2688 or shape[0] < 2:
        raise ValueError(
            "H3 E-grid has an unexpected shape. Expected [N>=2, 2688], "
            f"got {shape}."
        )



def _candidate_object_paths(module_name: str):
    """
    Produce possible ModelPatcher object paths for an AdaLN LoRA module.

    LoRA files in the wild may store names as either:
      diffusion_model.blocks.0.adaln_proj.linear
    or:
      blocks.0.adaln_proj.linear

    ComfyUI ModelPatcher roots have also changed across versions/custom loaders:
    some patchers root at BaseModel (needs 'diffusion_model.'), while others root
    directly at MiniMaxH3Model (must NOT have 'diffusion_model.').
    """
    base = module_name.rsplit(".linear", 1)[0]
    out = []

    def add(x):
        if x and x not in out:
            out.append(x)

    add(base)
    if base.startswith("diffusion_model."):
        add(base[len("diffusion_model."):])
    else:
        add("diffusion_model." + base)

    # Defensive normalization for accidentally doubled prefixes.
    while base.startswith("diffusion_model.diffusion_model."):
        base = base[len("diffusion_model."):]
        add(base)
        add(base[len("diffusion_model."):])

    return out


def _resolve_model_object_path(model_patcher, module_name: str):
    errors = []
    for candidate in _candidate_object_paths(module_name):
        try:
            obj = model_patcher.get_model_object(candidate)
            return candidate, obj
        except Exception as exc:
            errors.append(f"{candidate!r}: {type(exc).__name__}: {exc}")

    raise AttributeError(
        "Could not resolve AdaLN module against this ComfyUI ModelPatcher root. "
        f"LoRA module={module_name!r}. Tried: " + " | ".join(errors)
    )



def _payload_runtime_info(payload):
    if not isinstance(payload, dict):
        return {
            "type": type(payload).__name__,
            "keys": None,
            "segments": None,
        }

    layout = payload.get("layout")
    segments = None
    if layout is not None:
        try:
            segments = [(a, b, kind) for a, b, kind in layout.segments]
        except Exception:
            segments = f"<unreadable {type(layout).__name__}.segments>"

    return {
        "type": type(payload).__name__,
        "keys": sorted(str(k) for k in payload.keys()),
        "segments": segments,
    }


def _conditioning_flags(payload):
    """
    Derive visual/audio conditioning flags from current H3 packed-layout metadata.

    Current ComfyUI uses layout segment kinds:
      cond / ref_img   -> visual conditioning
      ref_audio        -> audio conditioning

    Older payloads may not expose layout, so we retain a conservative fallback.
    """
    has_vis = False
    has_aud = False
    vis_aug = 0.999
    aud_aug = 1.0

    if not isinstance(payload, dict):
        return has_vis, has_aud, vis_aug, aud_aug

    try:
        vis_aug = float(payload.get("visual_cond_noise_aug", 0.999))
    except Exception:
        vis_aug = 0.999
    try:
        aud_aug = float(payload.get("audio_cond_noise_aug", 1.0))
    except Exception:
        aud_aug = 1.0

    layout = payload.get("layout")
    if layout is not None:
        try:
            kinds = [kind for _, _, kind in layout.segments]
            has_vis = any(k in ("cond", "ref_img") for k in kinds)
            has_aud = any(k == "ref_audio" for k in kinds)
            return has_vis, has_aud, vis_aug, aud_aug
        except Exception:
            pass

    # Legacy fallback. Keyframes are visual. References can be heterogeneous;
    # inspect reference kind when available instead of assuming every ref is visual.
    has_vis = bool(payload.get("keyframes"))
    refs = payload.get("refs") or ()
    try:
        for ref in refs:
            if not isinstance(ref, dict):
                has_vis = True
                continue
            kind = str(ref.get("kind", "")).lower()
            if kind in ("audio", "ref_audio"):
                has_aud = True
            elif kind in ("video_audio", "audio_video"):
                has_vis = True
                has_aud = True
            else:
                has_vis = True
    except Exception:
        has_vis = has_vis or bool(refs)

    return has_vis, has_aud, vis_aug, aud_aug


def _core_compatible_unique_t(backend, timestep, shift_v, shift_a, payload):
    """
    Fallback implementation matching current ComfyUI H3 unique modulation rows.
    Used only when the installed backend helper has an unknown signature.
    """
    if not hasattr(backend, "_time_shift_sigma"):
        raise RuntimeError(
            "H3 Turbo backend has an unsupported _unique_t signature and no "
            "_time_shift_sigma helper for the compatibility fallback."
        )

    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - backend._time_shift_sigma(sv, shift_v, shift_a)

    has_vis, has_aud, vis_aug, aud_aug = _conditioning_flags(payload)
    values = {t_v, t_a}
    if has_vis:
        values.add(max(t_v, vis_aug))
    if has_aud:
        values.add(max(t_a, aud_aug))
    return sorted(values), "core-compatible fallback"


def _call_backend_unique_t(backend, timestep, shift_v, shift_a, payload):
    """
    The H3 Turbo backend has used more than one _unique_t API across revisions.

    Supported forms include:
      _unique_t(timestep, shift_v, shift_a, has_vis_cond)
      _unique_t(timestep, shift_v, shift_a, payload)
      _unique_t(timestep, shift_v, shift_a,
                has_vis_cond, has_aud_cond, vis_aug, aud_aug)

    We inspect the installed function instead of assuming one revision.
    """
    fn = getattr(backend, "_unique_t", None)
    if fn is None:
        return _core_compatible_unique_t(
            backend, timestep, shift_v, shift_a, payload
        )

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    names = [p.name for p in params]

    has_vis, has_aud, vis_aug, aud_aug = _conditioning_flags(payload)

    # Current/newer payload-aware backend revisions.
    if len(params) == 4:
        fourth = names[3].lower()
        if "payload" in fourth or fourth in ("minimax_payload", "h3_payload"):
            return (
                fn(timestep, shift_v, shift_a, payload),
                f"H3 backend _unique_t{sig} [payload-aware]",
            )
        return (
            fn(timestep, shift_v, shift_a, has_vis),
            f"H3 backend _unique_t{sig} [legacy visual-cond]",
        )

    # Transitional variants that added audio-condition awareness.
    if len(params) == 5:
        return (
            fn(timestep, shift_v, shift_a, has_vis, has_aud),
            f"H3 backend _unique_t{sig} [visual+audio-cond]",
        )

    # Issue/PR variants may expose visual/audio augmentation values too.
    if len(params) >= 7:
        return (
            fn(
                timestep,
                shift_v,
                shift_a,
                has_vis,
                has_aud,
                vis_aug,
                aud_aug,
            ),
            f"H3 backend _unique_t{sig} [visual+audio+augmentation]",
        )

    # Unknown revision: use current ComfyUI-equivalent math rather than guessing
    # a positional API.
    return _core_compatible_unique_t(
        backend, timestep, shift_v, shift_a, payload
    )



def _curve_aligned_egrid_rows(t_emb, table, egrid, shared, out_dtype):
    """Recover the *fractional* curve position represented by the model's
    actual 8-D curve embedding, then interpolate the matching 2688-D E-grid.

    This is intentionally based on the t_emb that MiniMaxH3Model really passes
    to AdaLN rather than on a separately reconstructed unique_t ordering.  It
    therefore stays aligned with ComfyUI's own modulation rows for ref_img,
    ref_audio, and mixed reference layouts.

    Unlike a nearest-grid-row lookup, this projects onto every adjacent segment
    of adaln_t_table and preserves the fractional coordinate between rows.
    The expensive alignment is done once per diffusion-model forward and shared
    by all 51 AdaLN modules.
    """
    serial = int(shared.get("forward_serial", -1))
    device = t_emb.device
    cache_key = (serial, str(device), str(out_dtype), int(t_emb.shape[0]))
    cached = shared.get("aligned_cache")
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"], cached["positions"], cached["max_residual"]

    # Keep the tiny curve table and the ~11 MiB full E-grid resident on the
    # active device.  This is safe compared with caching all model weights and
    # avoids repeatedly copying E for every step/module.
    dev_key = str(device)
    dev_cache = shared.setdefault("curve_device_cache", {})
    dc = dev_cache.get(dev_key)
    if dc is None:
        tb = table.detach().to(device=device, dtype=torch.float32)
        eg = egrid.detach().to(device=device, dtype=torch.float32)
        dc = (tb, eg)
        dev_cache[dev_key] = dc
    else:
        tb, eg = dc

    if tb.ndim != 2 or eg.ndim != 2 or tb.shape[0] != eg.shape[0] or tb.shape[0] < 2:
        raise RuntimeError(
            "AdaLN curve/E-grid shape mismatch during row alignment: "
            f"table={tuple(tb.shape)}, egrid={tuple(eg.shape)}"
        )
    if t_emb.ndim != 2 or t_emb.shape[1] != tb.shape[1]:
        raise RuntimeError(
            "AdaLN curve width mismatch during row alignment: "
            f"t_emb={tuple(t_emb.shape)}, table={tuple(tb.shape)}"
        )

    y = t_emb.detach().to(device=device, dtype=torch.float32)
    p0 = tb[:-1]                         # [N-1, 8]
    d = tb[1:] - p0                      # [N-1, 8]
    denom = (d * d).sum(dim=1).clamp_min_(1e-20)  # [N-1]

    # M is normally only 2-4 rows, so this is tiny (~M*1024*8) and happens once
    # per diffusion step, not once per AdaLN module.
    rel = y[:, None, :] - p0[None, :, :]
    alpha = ((rel * d[None, :, :]).sum(dim=2) / denom[None, :]).clamp_(0.0, 1.0)
    recon = p0[None, :, :] + alpha[:, :, None] * d[None, :, :]
    err = ((recon - y[:, None, :]) ** 2).sum(dim=2)
    idx = err.argmin(dim=1)
    row = torch.arange(y.shape[0], device=device)
    frac = alpha[row, idx]
    best_err = err[row, idx]

    # Same fractional coordinate on the bundled full-width SiLU(time-embed)
    # curve.  This is the full 2688-D row the LoRA was trained against.
    full = torch.lerp(eg[idx], eg[idx + 1], frac[:, None]).to(dtype=out_dtype)
    positions = (idx.to(torch.float32) + frac).detach()
    max_residual = float(best_err.max().item()) if best_err.numel() else 0.0

    shared["aligned_cache"] = {
        "key": cache_key,
        "rows": full,
        "positions": positions,
        "max_residual": max_residual,
    }
    return full, positions, max_residual


def _make_v63_adaln_forward(base, a, b, shared, table, egrid, cache_weights=True):
    """AdaLN forward patch with table-aligned E-grid rows and lazy GPU LoRA cache."""
    weight_cache = {}
    state = {"warned_align": False, "warned_shape": False, "logged_cache": False}

    def _weights(device, dtype):
        if not cache_weights:
            return (
                a.to(device=device, dtype=dtype),
                b.to(device=device, dtype=dtype),
            )
        key = (str(device), str(dtype))
        hit = weight_cache.get(key)
        if hit is not None:
            return hit
        av = a.to(device=device, dtype=dtype)
        bv = b.to(device=device, dtype=dtype)
        weight_cache[key] = (av, bv)
        if not shared.get("logged_weight_cache"):
            _info(
                "AdaLN lazy device cache active: A/B tensors are transferred once "
                "per AdaLN module/device/dtype and reused on later diffusion steps."
            )
            shared["logged_weight_cache"] = True
        return av, bv

    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = None

        # Curve-mode checkpoints expose the actual 8-D t_emb. Align from that
        # tensor directly instead of trusting independently reconstructed row order.
        if table is not None and egrid is not None and not base.apply_silu:
            try:
                st, positions, residual = _curve_aligned_egrid_rows(
                    t_emb, table, egrid, shared, x.dtype
                )
                if not shared.get("logged_alignment"):
                    pos_preview = [round(float(v), 4) for v in positions.detach().cpu().tolist()]
                    _info(
                        "AdaLN table alignment first-forward OK: "
                        f"rows={int(st.shape[0])}; grid_positions={pos_preview}; "
                        f"max_curve_residual={residual:.3e}; mode=fractional-segment"
                    )
                    shared["logged_alignment"] = True
            except Exception as exc:
                if not state["warned_align"]:
                    _warn(
                        "AdaLN table alignment failed; falling back to legacy "
                        f"unique_t E-grid interpolation for this run: {type(exc).__name__}: {exc}"
                    )
                    state["warned_align"] = True
                st = None

        # Compatibility fallback for unusual backend/ComfyUI revisions.
        if st is None:
            serial = int(shared.get("forward_serial", -1))
            fb = shared.get("fallback_cache")
            if fb is not None and fb.get("serial") == serial and fb.get("dtype") == str(x.dtype):
                st = fb["rows"]
            else:
                us = shared.get("fallback_unique_t")
                backend = shared.get("backend")
                e = shared.get("egrid")
                if us is not None and backend is not None and e is not None:
                    st = backend._interp_egrid(us, e, x.device, x.dtype)
                    shared["fallback_cache"] = {
                        "serial": serial,
                        "dtype": str(x.dtype),
                        "rows": st,
                    }

        if st is not None and st.shape[0] == x.shape[0]:
            av, bv = _weights(x.device, x.dtype)
            sv = st.to(device=x.device, dtype=x.dtype)
            x = x + (bv @ (av @ sv.T)).T
        elif st is not None and not state["warned_shape"]:
            _warn(
                "AdaLN delta row count does not match model modulation rows; "
                f"delta_rows={int(st.shape[0])}, model_rows={int(x.shape[0])}. "
                "Skipping AdaLN delta for this module/step rather than misaligning rows."
            )
            state["warned_shape"] = True

        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


def _make_compatible_adaln_injector(backend):
    """
    Return a compatible _inject_adaln_egrid implementation that resolves
    AdaLN object paths dynamically instead of assuming ModelPatcher is rooted at
    a BaseModel containing `.diffusion_model`.

    v6.3 keeps the backend's bundled full-width E-grid and LoRA tensors, but aligns
    each runtime AdaLN row from the model's actual curve t_emb against
    adaln_t_table before applying the LoRA delta. Object-path resolution remains
    dynamic for ComfyUI compatibility.
    """
    required = (
        "_egrid",
        "_unique_t",
        "_interp_egrid",
    )
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise RuntimeError(
            "H3 Turbo backend is missing helpers required for the compatibility "
            f"injector: {missing}. Update the H3 Full-AdaLN loader/backend."
        )

    try:
        import comfy.patcher_extension
    except Exception as exc:
        raise RuntimeError(
            f"Could not import comfy.patcher_extension: {exc}"
        ) from exc

    shift_v_default = float(getattr(backend, "SHIFT_V", 12.0))
    shift_a_default = float(getattr(backend, "SHIFT_A", 3.0))

    def compat_inject(new_model, dm, lora, adaln, strength, cache_weights=True):
        E = backend._egrid()

        # Prefer the model's own curve table.  v6.3 aligns the LoRA's full-width
        # E-grid against the actual 8-D t_emb rows seen by AdaLN.
        table = getattr(dm, "adaln_t_table", None)
        if table is None:
            try:
                for _n, _t in list(dm.named_buffers()) + list(dm.named_parameters()):
                    if _n.endswith("adaln_t_table"):
                        table = _t
                        break
            except Exception:
                table = None
        if table is not None and (table.ndim != 2 or table.shape[0] != E.shape[0]):
            _warn(
                "adaln_t_table cannot be paired with the H3 E-grid; falling back "
                f"to legacy unique_t alignment. table={tuple(table.shape)}, "
                f"egrid={tuple(E.shape)}"
            )
            table = None

        shared = {
            "silu_temb": None,
            "backend": backend,
            "egrid": E,
            "fallback_unique_t": None,
            "forward_serial": 0,
            "aligned_cache": None,
            "fallback_cache": None,
            "logged_alignment": False,
        }
        fallback_shift_v = float(getattr(dm, "sigma_shift_video", shift_v_default))
        fallback_shift_a = float(getattr(dm, "sigma_shift_audio", shift_a_default))

        _info(
            "Using ComfyUI-compatible AdaLN object-path resolver; "
            "v6.3 table-aligned fractional E-grid path="
            f"{'enabled' if table is not None else 'fallback-only'}; "
            "runtime transformer_options shifts take precedence "
            f"(diffusion-model fallback video={fallback_shift_v}, "
            f"audio={fallback_shift_a})"
        )
        _info(
            "AdaLN A/B lazy GPU cache="
            f"{'enabled' if cache_weights else 'disabled'}"
        )

        runtime_state = {"logged_success": False}

        def wrap(executor, *args, **kwargs):
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            ctx = args[2] if len(args) > 2 else kwargs.get("context")
            transformer_options = (
                args[3] if len(args) > 3 else kwargs.get("transformer_options")
            )
            if not isinstance(transformer_options, dict):
                transformer_options = {}

            shift_v = float(
                transformer_options.get(
                    "minimax_h3_sigma_shift_video", fallback_shift_v
                )
            )
            shift_a = float(
                transformer_options.get(
                    "minimax_h3_sigma_shift_audio", fallback_shift_a
                )
            )
            shift_source = (
                "transformer_options"
                if (
                    "minimax_h3_sigma_shift_video" in transformer_options
                    or "minimax_h3_sigma_shift_audio" in transformer_options
                )
                else "diffusion_model_fallback"
            )

            payload = kwargs.get("minimax_payload")
            if payload is None:
                payload = {}

            try:
                if ts is None:
                    raise RuntimeError(
                        "AdaLN runtime injector did not receive `timestep`; cannot "
                        "reconstruct the full-width H3 time embedding."
                    )
                if ctx is None:
                    raise RuntimeError(
                        "AdaLN runtime injector did not receive `context`; cannot "
                        "select device/dtype for the reconstructed time embedding."
                    )
                if not isinstance(payload, dict):
                    raise TypeError(
                        "MiniMax-H3 `minimax_payload` must be a dict at the "
                        f"DIFFUSION_MODEL wrapper, got {type(payload).__name__}."
                    )

                us, unique_t_mode = _call_backend_unique_t(
                    backend, ts, shift_v, shift_a, payload
                )

                if not isinstance(us, (list, tuple)) or len(us) < 2:
                    raise RuntimeError(
                        f"_unique_t returned an invalid value: {us!r}"
                    )

                # Store the compatibility unique_t values, but do not eagerly
                # interpolate the 2688-D grid in the normal v6.3 path. The first
                # AdaLN module aligns from the model's actual 8-D t_emb and all
                # remaining modules reuse that result.
                shared["forward_serial"] = int(shared.get("forward_serial", 0)) + 1
                shared["fallback_unique_t"] = us
                shared["aligned_cache"] = None
                shared["fallback_cache"] = None

                if not runtime_state["logged_success"]:
                    info = _payload_runtime_info(payload)
                    _info(
                        "AdaLN runtime first-forward OK: "
                        f"effective_video_shift={shift_v}; "
                        f"effective_audio_shift={shift_a}; "
                        f"shift_source={shift_source}; "
                        f"unique_t_mode={unique_t_mode}; "
                        f"unique_rows={len(us)}; "
                        f"payload_keys={info['keys']}; "
                        f"layout_segments={info['segments']}"
                    )
                    runtime_state["logged_success"] = True

                return executor(*args, **kwargs)

            except Exception as exc:
                info = _payload_runtime_info(payload)
                try:
                    unique_sig = inspect.signature(backend._unique_t)
                except Exception:
                    unique_sig = "<unavailable>"

                _error(
                    "AdaLN runtime wrapper failed before/while reconstructing "
                    f"time embeddings: {type(exc).__name__}: {exc}"
                )
                _error(
                    "Runtime diagnostic: "
                    f"effective_video_shift={shift_v}, "
                    f"effective_audio_shift={shift_a}, "
                    f"shift_source={shift_source}, "
                    f"payload_type={info['type']}, "
                    f"payload_keys={info['keys']}, "
                    f"layout_segments={info['segments']}, "
                    f"H3 backend _unique_t signature={unique_sig}"
                )
                _error("Runtime traceback follows:")
                traceback.print_exc()
                raise

        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "h3_full_adaln_compat",
            wrap,
        )

        resolved = []
        for idx, name in enumerate(adaln):
            a_key = name + ".lora_A.weight"
            b_key = name + ".lora_B.weight"
            if a_key not in lora or b_key not in lora:
                raise KeyError(
                    f"AdaLN pair disappeared after preflight: {name}"
                )

            a = lora[a_key]
            b = lora[b_key] * strength

            path, target = _resolve_model_object_path(new_model, name)
            new_model.add_object_patch(
                path + ".forward",
                _make_v63_adaln_forward(target, a, b, shared, table, E, cache_weights=cache_weights),
            )
            resolved.append((name, path))

            if idx < 3:
                _info(f"AdaLN path resolved: {name} -> {path}")

        if len(resolved) > 3:
            _info(
                f"AdaLN path resolver patched {len(resolved)} modules "
                f"(showing first 3 above)."
            )

        return resolved

    return compat_inject


def _backend_needs_path_compat(model, report):
    """
    Detect whether the backend's fixed `diffusion_model.*` lookup is incompatible
    with the active ModelPatcher root. We test the first AdaLN module without
    mutating the model.
    """
    if not report["adaln"]:
        return False

    sample = report["adaln"][0]

    # If one of our dynamic candidates resolves but the backend's historical fixed
    # path does not, enable the compatibility injector.
    dynamic_path, _ = _resolve_model_object_path(model, sample)

    base = sample.rsplit(".linear", 1)[0]
    # Historical backend variants have used this assumption.
    backend_path = "diffusion_model." + base
    try:
        model.get_model_object(backend_path)
        backend_path_ok = True
    except Exception:
        backend_path_ok = False

    if not backend_path_ok:
        _warn(
            "H3 backend AdaLN object-path assumption does not match this ComfyUI "
            f"ModelPatcher root. Resolved sample as {dynamic_path!r}; enabling "
            "compatibility injector."
        )
        return True

    # Even when the backend-constructed path resolves, a LoRA name already carrying
    # `diffusion_model.` can produce a doubled prefix in some backend revisions.
    if sample.startswith("diffusion_model.") and backend_path.startswith(
        "diffusion_model.diffusion_model."
    ):
        _warn(
            "LoRA module names already include 'diffusion_model.'; enabling "
            "compatibility injector to prevent a doubled prefix."
        )
        return True

    return False


def _normalize_lora_tensor_keys(lora: dict):
    """
    Normalize Comfy-style H3 LoRA tensor names for backend helpers.

    The backend backbone helpers expect module names such as:
        blocks.0.attn.qkv_proj
    and then construct the Comfy ModelPatcher key:
        diffusion_model.blocks.0.attn.qkv_proj.weight

    Some experimental H3 LoRAs already store:
        diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight

    Passing those names through unchanged makes the backend construct:
        diffusion_model.diffusion_model.blocks....
    which silently matches zero backbone modules.  We strip exactly one leading
    `diffusion_model.` namespace before using the backend backbone helpers.

    Returns (normalized_dict, changed_count).
    """
    out = {}
    changed = 0
    prefix = "diffusion_model."

    for key, value in lora.items():
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        if new_key != key:
            changed += 1
        if new_key in out:
            raise ValueError(
                "LoRA key normalization produced a collision: "
                f"{key!r} -> {new_key!r}. Refusing to continue."
            )
        out[new_key] = value

    return out, changed


def _choose_bypass_canary_module(bypass_modules):
    """
    Prefer a qkv projection as a lightweight runtime canary.  Fall back
    to the first bypassed module if the LoRA layout differs.
    """
    for name in bypass_modules:
        if name.endswith("blocks.0.attn.qkv_proj") or name == "blocks.0.attn.qkv_proj":
            return name
    for name in bypass_modules:
        if name.endswith(".attn.qkv_proj"):
            return name
    return bypass_modules[0] if bypass_modules else None


def _add_backbone_runtime_verifier(backend, new_model, dm, bypass_modules, merged_modules):
    """
    Fail loudly if a run that *should* have live bypass LoRA adapters reaches the
    first diffusion-model forward with only the base Linear forward.

    This specifically prevents the v0.4 failure mode where 208 backbone modules
    were listed in the LoRA, but zero adapters/injections were actually attached.
    """
    try:
        import comfy.patcher_extension
    except Exception as exc:
        raise RuntimeError(
            f"Could not import comfy.patcher_extension for runtime verification: {exc}"
        ) from exc

    canary_name = _choose_bypass_canary_module(bypass_modules)
    state = {"checked": False}

    def wrap(executor, *args, **kwargs):
        if not state["checked"]:
            state["checked"] = True
            bypass_owner = None
            injected = getattr(new_model, "is_injected", None)

            if canary_name is not None:
                try:
                    module = backend.comfy.utils.get_attr(dm, canary_name)
                    bypass_owner = type(
                        getattr(module.forward, "__self__", None)
                    ).__name__
                except Exception as exc:
                    raise RuntimeError(
                        "Could not inspect backbone LoRA runtime canary "
                        f"{canary_name!r}: {type(exc).__name__}: {exc}"
                    ) from exc

                # Runtime truth for bypass mode is the live module.forward owner.
                # ComfyUI can have the BypassForwardHook installed and active
                # while ModelPatcher.is_injected remains False; that flag describes
                # injection lifecycle/state, not whether this bypass forward hook is
                # currently owning the module call. Treat it as diagnostic only.
                if bypass_owner != "BypassForwardHook":
                    raise RuntimeError(
                        "Backbone LoRA bypass is NOT active on first forward. "
                        f"canary={canary_name!r}, "
                        f"forward_owner={bypass_owner!r}, "
                        f"model.is_injected={injected!r}, "
                        f"expected_bypass_modules={len(bypass_modules)}, "
                        f"merged_modules={len(merged_modules)}. "
                        "Generation aborted instead of silently running the base "
                        "backbone."
                    )

            _info(
                "Backbone runtime first-forward OK: "
                f"bypass_modules={len(bypass_modules)}, "
                f"merged_modules={len(merged_modules)}, "
                f"canary={canary_name!r}, "
                f"forward_owner={bypass_owner!r}, "
                f"is_injected={injected!r}"
            )

        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "h3_full_adaln_backbone_verify",
        wrap,
    )


def _apply_normalized_h3_lora(
    backend,
    model,
    path,
    backbone_strength,
    adaln_strength,
    low_vram,
    pruned,
    adaln_gpu_cache=True,
    apply_mode="full",
):
    """
    Apply the LoRA using the H3 Turbo backend helper implementations after normalizing the
    tensor namespace.

    We intentionally do not call MiniMaxH3TurboLoRA.apply_lora here because that
    method derives module names directly from the file keys.  For a file whose
    keys already begin with `diffusion_model.`, the backend then prepends another
    `diffusion_model.` and silently attaches zero backbone adapters.

    Backbone numerical operations remain backend-provided:
      - _apply_bypass_lora
      - _apply_merge_lora
      - _int8_fused_fc2

    Curve-mode AdaLN uses the same H3 E-grid/LoRA tensors with v6.3's
    table-aligned fractional runtime injector.
    """
    required = (
        "_apply_bypass_lora",
        "_apply_merge_lora",
        "_int8_fused_fc2",
    )
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise RuntimeError(
            "Installed H3 Turbo backend is missing helper functions required for "
            f"safe namespace-normalized loading: {missing}"
        )

    allowed_modes = {"full", "adaln_only", "backbone_only"}
    if apply_mode not in allowed_modes:
        raise ValueError(
            f"Unknown apply_mode={apply_mode!r}. Expected one of {sorted(allowed_modes)}."
        )

    raw_lora = backend.comfy.utils.load_torch_file(path, safe_load=True)
    if not isinstance(raw_lora, dict) or not raw_lora:
        raise ValueError(
            f"LoRA file loaded as {type(raw_lora).__name__} and contains no tensors."
        )

    lora, changed = _normalize_lora_tensor_keys(raw_lora)
    _info(
        "LoRA namespace normalization: "
        f"stripped leading 'diffusion_model.' from {changed}/{len(raw_lora)} tensors."
    )

    modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})
    if not modules:
        raise ValueError("No LoRA modules remained after namespace normalization.")

    dm = model.model.diffusion_model
    new_model = model.clone()

    if pruned:
        backbone = [m for m in modules if "adaln_proj" not in m]
        adaln = [m for m in modules if "adaln_proj" in m]
    else:
        backbone, adaln = modules, []

    _info(
        "Normalized module map: "
        f"total={len(modules)}, backbone={len(backbone)}, adaln={len(adaln)}"
    )

    requested_backbone = apply_mode in ("full", "backbone_only")
    requested_adaln = apply_mode in ("full", "adaln_only")

    _info(
        "Apply mode: "
        f"{apply_mode} (backbone={'on' if requested_backbone else 'off'}, "
        f"adaln={'on' if requested_adaln else 'off'})"
    )
    _info(
        "Effective strengths: "
        f"backbone={float(backbone_strength):.4f}, "
        f"adaln={float(adaln_strength):.4f}"
    )

    bypass_modules = []
    merged_modules = []

    if requested_backbone and low_vram:
        n_merged = backend._apply_merge_lora(
            new_model, lora, backbone, float(backbone_strength)
        )
        merged_modules = list(backbone)

        if n_merged != len(backbone):
            raise RuntimeError(
                "Backbone merge count mismatch: "
                f"expected={len(backbone)}, applied={n_merged}. "
                "Refusing a partial LoRA load."
            )

        _info(f"Backbone merge attached {n_merged}/{len(backbone)} modules.")

    elif requested_backbone:
        fc2_fused = set(backend._int8_fused_fc2(dm, backbone))
        bypass_modules = [m for m in backbone if m not in fc2_fused]
        merged_modules = sorted(fc2_fused)

        n_bypass = backend._apply_bypass_lora(
            new_model, lora, bypass_modules, float(backbone_strength)
        )
        n_merged = 0
        if merged_modules:
            n_merged = backend._apply_merge_lora(
                new_model, lora, merged_modules, float(backbone_strength)
            )

        if n_bypass != len(bypass_modules):
            raise RuntimeError(
                "Backbone bypass count mismatch: "
                f"expected={len(bypass_modules)}, attached={n_bypass}. "
                "This usually means the LoRA/model key namespace still does not "
                "match. Refusing a partial LoRA load."
            )
        if n_merged != len(merged_modules):
            raise RuntimeError(
                "INT8-fused FC2 merge count mismatch: "
                f"expected={len(merged_modules)}, applied={n_merged}. "
                "Refusing a partial LoRA load."
            )

        injections = new_model.injections.get("bypass_lora", [])
        if bypass_modules and not injections:
            raise RuntimeError(
                "H3 backend reported bypass modules but created zero bypass_lora "
                "injections. Refusing to continue."
            )

        _info(
            "Backbone patches validated: "
            f"{n_bypass} bypass adapters, {len(injections)} injections, "
            f"{n_merged} INT8-fused/merged modules; "
            f"total={n_bypass + n_merged}/{len(backbone)}."
        )
    else:
        _info(
            "Backbone patch application skipped by apply_mode. "
            f"Available backbone modules={len(backbone)}."
        )

    # v0.6.26 keeps v0.6.25's native AdaLN handoff step, then runs the bundled
    # PlagueKind port before returning from this same node. Attach only the raw
    # AdaLN part through ComfyUI's own LoRA parser so the port core sees exactly
    # the same mismatched LoRAAdapter patches it would see after a normal
    # LoraLoaderModelOnly. The custom H3 routing above remains responsible only
    # for the 208 non-AdaLN backbone modules, so there is no double application.
    native_adaln_attached = 0
    native_adaln_loaded = 0
    if pruned and adaln:
        import comfy.lora
        import comfy.lora_convert

        raw_adaln = {
            key: value for key, value in raw_lora.items()
            if "adaln_proj" in key
        }
        if not raw_adaln:
            raise RuntimeError(
                "LoRA header reported AdaLN modules, but no raw AdaLN tensors "
                "were found for native ComfyUI attachment."
            )

        key_map = {}
        key_map = comfy.lora.model_lora_keys_unet(new_model.model, key_map)
        converted_adaln = comfy.lora_convert.convert_lora(raw_adaln)
        loaded_adaln = comfy.lora.load_lora(converted_adaln, key_map)

        native_adaln_loaded = len(loaded_adaln)
        if native_adaln_loaded != len(adaln):
            raise RuntimeError(
                "Native AdaLN LoRA parse count mismatch: "
                f"expected={len(adaln)}, parsed={native_adaln_loaded}. "
                "Refusing a partial AdaLN handoff."
            )

        attached_keys = new_model.add_patches(
            loaded_adaln, float(adaln_strength)
        )
        native_adaln_attached = len(attached_keys)
        if native_adaln_attached != len(adaln):
            raise RuntimeError(
                "Native AdaLN patch attach count mismatch: "
                f"expected={len(adaln)}, attached={native_adaln_attached}. "
                "Refusing a partial AdaLN handoff."
            )

        _info(
            "AdaLN handoff ready: "
            f"{native_adaln_attached}/{len(adaln)} native ComfyUI LoRA patches "
            f"attached at strength={float(adaln_strength):.4f}; "
            "integrated AdaLN fix will run before this node returns."
        )
    elif pruned and not adaln:
        _warn(
            "Pruned/curve H3 detected, but this LoRA exposes no AdaLN module "
            "pairs. Backbone loading can continue without an AdaLN handoff."
        )
    else:
        _info(
            "Model is not a pruned/curve H3; no pruned AdaLN handoff is needed."
        )

    # v6.3 omits the backend per-step debug-print wrapper. Our own
    # first-forward verifier still proves bypass ownership, while avoiding an
    # extra Python wrapper + console print on every diffusion step.
    _info("Backend per-step debug wrapper disabled; first-forward verifier remains active.")

    # Our verifier turns the previously-silent BASE ONLY condition into a hard
    # runtime failure on the first model forward, but only when backbone LoRA is
    # intentionally enabled for this diagnostic mode.
    if requested_backbone:
        _add_backbone_runtime_verifier(
            backend,
            new_model,
            dm,
            bypass_modules=bypass_modules,
            merged_modules=merged_modules,
        )
    else:
        _info(
            "Backbone runtime verifier skipped because apply_mode disables "
            "backbone LoRA on purpose."
        )

    try:
        p0 = dm.blocks[0].attn.qkv_proj.weight
        wdt, wdev = str(p0.dtype), str(p0.device)
    except Exception:
        wdt, wdev = "?", "?"

    _info(
        "SAFE APPLY COMPLETE: "
        f"mode={apply_mode}, "
        f"backbone={len(backbone)} "
        f"(enabled={'yes' if requested_backbone else 'no'}, "
        f"strength={float(backbone_strength):.4f}, "
        f"bypass={len(bypass_modules)}, merged={len(merged_modules)}), "
        f"adaln_native_handoff={native_adaln_attached}/{len(adaln)} "
        f"(strength={float(adaln_strength):.4f}), "
        f"model={type(new_model.model).__name__}, "
        f"weight_dtype={wdt}, weight_dev={wdev}"
    )

    return (new_model,)


_ADALN_LOG = logging.getLogger("H3AdaLN")
_ADALN_FIX_MODES = ("port", "strip", "off")


def _apply_integrated_adaln_fix(model, mode: str):
    """Run the bundled PlagueKind AdaLN fix with the same fail-open semantics.

    The v6.25 handoff intentionally attaches the raw mismatched AdaLN LoRA patches
    first.  This helper performs the same ``fix_model`` operation as the standalone
    H3 AdaLN LoRA Fix node, but immediately inside this loader so no second node is
    needed in the graph.
    """
    mode = str(mode)
    if mode not in _ADALN_FIX_MODES:
        raise ValueError(
            f"Unknown adaln_fix_mode={mode!r}; expected one of {_ADALN_FIX_MODES}."
        )

    try:
        patched, report = _adaln_patch.fix_model(model, mode)
    except Exception:  # match the upstream standalone node's fail-open behavior
        _ADALN_LOG.exception(
            "[H3AdaLN] Integrated AdaLN LoRA fix failed; passing the model "
            "through unchanged."
        )
        return model, None

    _ADALN_LOG.info("[H3AdaLN] %s", _adaln_patch.format_report(report))
    for note in report["notes"][:8]:
        _ADALN_LOG.warning("[H3AdaLN]   %s", note)
    if len(report["notes"]) > 8:
        _ADALN_LOG.warning(
            "[H3AdaLN]   ... and %d more", len(report["notes"]) - 8
        )

    _info(
        "Integrated AdaLN fix complete: "
        f"mode={mode}, effective_mode={report.get('effective_mode')}, "
        f"ported={report.get('ported', 0)}, stripped={report.get('stripped', 0)}, "
        f"keys={report.get('keys', 0)}, residual={report.get('residual')}"
    )
    return patched, report


class SilverOxidesH3FullAdaLNLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "adaln_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "backbone_strength": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "adaln_fix_mode": (
                    list(_ADALN_FIX_MODES),
                    {"default": "port"},
                ),
                "console_log": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "console log on",
                        "label_off": "console log off",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "MiniMaxH3"
    DESCRIPTION = (
        "MiniMax-H3 Turbo LoRA bridge with direct AdaLN/Backbone strengths and "
        "the PlagueKind H3 AdaLN LoRA Fix integrated into the same node. "
        "adaln_fix_mode=port rebases dense AdaLN LoRA patches onto pruned curve "
        "H3; strip removes incompatible AdaLN patches; off leaves them untouched. "
        "No separate H3 AdaLN LoRA Fix node is required."
    )

    def load_lora(
        self,
        model,
        lora_name,
        adaln_strength=1.0,
        backbone_strength=0.45,
        adaln_fix_mode="port",
        console_log=False,
    ):
        global _LOG_ENABLED
        _LOG_ENABLED = bool(console_log)

        adaln_strength = float(adaln_strength)
        backbone_strength = float(backbone_strength)
        adaln_fix_mode = str(adaln_fix_mode)
        low_vram = False
        apply_mode = "full"

        _info(f"v{VERSION} H3 Turbo LoRA Bridge")
        _info(
            f"Preset: adaln_strength={adaln_strength:.2f}, "
            f"backbone_strength={backbone_strength:.2f}, "
            f"adaln_fix_mode={adaln_fix_mode}, "
            "custom backbone + native AdaLN handoff + integrated fix"
        )

        try:
            dm, pruned = _inspect_model(model)

            path = folder_paths.get_full_path("loras", lora_name)
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(
                    f"LoRA '{lora_name}' could not be resolved in ComfyUI/models/loras."
                )

            size_mb = os.path.getsize(path) / (1024 * 1024)
            _info(f"LoRA: {path} ({size_mb:.1f} MiB)")

            report = _inspect_lora_header(path)
            _info(
                "LoRA header: "
                f"tensors={len(report['keys'])}, paired_modules={len(report['paired'])}, "
                f"backbone_pairs={len(report['backbone'])}, adaln_pairs={len(report['adaln'])}"
            )
            _validate_adaln_layout(report, pruned)

            if pruned and not report["adaln"]:
                _warn(
                    "Pruned/curve H3 detected, but this LoRA has no AdaLN A/B pairs. "
                    "Only ordinary backbone patches will be applied."
                )
            elif pruned:
                sample = report["adaln"][0]
                _info(
                    "Full-width AdaLN layout validated. Example: "
                    f"{sample}: A={report['shapes'][sample + '.lora_A.weight']}, "
                    f"B={report['shapes'][sample + '.lora_B.weight']}"
                )

            backend = _load_backend_module()
            _install_h3_memory_fc1_bridge(backend)

            result = _apply_normalized_h3_lora(
                backend=backend,
                model=model,
                path=path,
                backbone_strength=backbone_strength,
                adaln_strength=adaln_strength,
                low_vram=low_vram,
                pruned=pruned,
                adaln_gpu_cache=False,
                apply_mode=apply_mode,
            )

            if not isinstance(result, tuple) or len(result) != 1:
                raise RuntimeError(
                    "Backend returned an unexpected result. "
                    f"Expected a one-item tuple (MODEL,), got {type(result)!r}: {result!r}"
                )
            if result[0] is None:
                raise RuntimeError("Backend returned MODEL=None.")

            fixed_model, fix_report = _apply_integrated_adaln_fix(
                result[0], adaln_fix_mode
            )

            if fix_report is not None and adaln_fix_mode == "port" and pruned and report["adaln"]:
                expected = len(report["adaln"])
                ported = int(fix_report.get("ported", 0))
                if fix_report.get("effective_mode") == "port" and ported != expected:
                    _warn(
                        "Integrated AdaLN port count differs from the LoRA header: "
                        f"expected={expected}, ported={ported}. Check the H3AdaLN notes above."
                    )

            _info(
                f"READY: adaln_strength={adaln_strength:.2f}, "
                f"backbone_strength={backbone_strength:.2f}, "
                f"adaln_fix_mode={adaln_fix_mode}"
            )
            return (fixed_model,)

        except Exception as exc:
            _error(f"{type(exc).__name__}: {exc}")
            _error("Full traceback follows:")
            traceback.print_exc()
            raise

NODE_CLASS_MAPPINGS = {
    "SilverOxidesH3FullAdaLNLoRALoader": SilverOxidesH3FullAdaLNLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverOxidesH3FullAdaLNLoRALoader":
        "H3 Turbo LoRA Bridge",
}
