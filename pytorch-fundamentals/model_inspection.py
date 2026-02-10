"""
activation_utils.py

A small, readable toolbox for inspecting CNNs during a forward pass:
1) Model inspection (parameter counts, parameter shapes, module tree)
2) Shape tracing (log input/output tensor shapes through key layers)
3) Activation logging (capture activations via hooks, compute stats, plot histograms)

Design goals:
- Minimal dependencies (torch + matplotlib)
- No need to modify your model.forward() for debugging
- Works well with modular blocks like ConvBlock / nn.Sequential
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =============================================================================
# 1) BASIC MODEL INSPECTION (simple, notebook-friendly helpers)
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count total parameters in a model.

    Why it helps:
      - sanity check model size
      - compare architectures quickly
      - avoid accidentally huge linear layers after flattening

    Args:
      model: any nn.Module
      trainable_only: if True, counts only parameters with requires_grad=True

    Returns:
      total number of parameters (int)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_named_parameters_shapes(model: nn.Module) -> None:
    """
    Print each parameter name + shape.

    Useful for:
      - diagnosing matmul errors like: "mat1 and mat2 shapes cannot be multiplied"
      - confirming Conv2d weight shapes: (out_channels, in_channels, kH, kW)
      - confirming Linear weight shapes: (out_features, in_features)
    """
    for name, param in model.named_parameters():
        print(f"{name}: {tuple(param.shape)}")


def print_top_level_children(model: nn.Module) -> None:
    """
    Print top-level modules directly under the root module.

    Useful for modular architectures:
      - features, classifier, etc.
    """
    for name, module in model.named_children():
        print(name, module)


def print_all_modules(model: nn.Module, skip_root: bool = True) -> None:
    """
    Print all modules including nested sub-modules.

    This is especially useful with nn.Sequential() and custom blocks
    because it reveals the names you can use in include/exclude filters.
    """
    for name, module in model.named_modules():
        if skip_root and name == "":
            continue
        print(name, module)


# =============================================================================
# 2) SHAPE TRACING (log tensor shapes through layers without editing forward)
# =============================================================================

class ShapeTracer:
    """
    A forward-hook based "shape logger".

    Why it helps:
      - You can trace shapes through Conv/Pool/Flatten/Linear without
        adding print() statements inside forward().
      - Great for debugging flatten dimension mistakes.

    Typical usage:
      tracer = ShapeTracer(model)
      tracer.register()
      tracer.clear()
      _ = model(x)
      tracer.print()
      tracer.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_types: Tuple[type, ...] = (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Flatten, nn.Linear),
    ):
        self.model = model
        self.layer_types = layer_types
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.records: List[Tuple[str, str, Tuple[int, ...]]] = []

    @staticmethod
    def _shape_of(obj: Any) -> Optional[Tuple[int, ...]]:
        """
        Hook inputs/outputs may be:
          - a Tensor
          - a tuple/list of Tensors
          - something else (rare)
        This returns the shape for the first Tensor-like object we can find.
        """
        if torch.is_tensor(obj):
            return tuple(obj.shape)
        if isinstance(obj, (tuple, list)) and len(obj) > 0 and torch.is_tensor(obj[0]):
            return tuple(obj[0].shape)
        return None

    def register(self) -> None:
        """Attach hooks to selected layer types."""
        self.remove()  # avoid duplicate hooks if register() is called twice

        for name, module in self.model.named_modules():
            if name == "":
                continue  # skip the root module itself
            if not isinstance(module, self.layer_types):
                continue

            def make_hook(layer_name: str):
                def hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any):
                    # Inputs is usually a tuple. We log the first tensor input if present.
                    in_shape = self._shape_of(inputs[0]) if len(inputs) > 0 else None
                    out_shape = self._shape_of(output)

                    if in_shape is not None:
                        self.records.append((layer_name, "in ", in_shape))
                    if out_shape is not None:
                        self.records.append((layer_name, "out", out_shape))
                return hook

            self.handles.append(module.register_forward_hook(make_hook(name)))

    def clear(self) -> None:
        """Clear stored shape records (call before each forward pass you want to inspect)."""
        self.records.clear()

    def remove(self) -> None:
        """Remove hooks (good practice when done)."""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def print(self) -> None:
        """Pretty print the recorded shape trace."""
        for name, io, shp in self.records:
            print(f"{name:35s} {io}: {shp}")


# =============================================================================
# 3) ACTIVATION LOGGING + STATS + HISTOGRAMS
# =============================================================================

@dataclass
class ActStats:
    """
    Minimal activation statistics.
    These are intentionally simple to build intuition:
      - mean/std: signal strength + stability
      - min/max: detect outliers or explosion
      - percentiles: robust view (less sensitive than max)
      - zero_frac: especially useful for ReLU deadness
    """
    layer: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float
    p5: float
    p50: float
    p95: float
    zero_frac: float


def _flatten_cpu_float(x: torch.Tensor) -> torch.Tensor:
    """
    Convert activation to a flat CPU float tensor for stable stats/plotting.
    Detach prevents autograd interference.
    """
    if not torch.is_floating_point(x):
        x = x.float()
    return x.detach().to("cpu", dtype=torch.float32).flatten()


def compute_stats(x: torch.Tensor, layer_name: str) -> ActStats:
    """Compute activation stats for a tensor activation."""
    flat = _flatten_cpu_float(x)

    # Robust percentiles help you see the "typical" range
    qs = torch.tensor([0.05, 0.50, 0.95], dtype=torch.float32)
    p5, p50, p95 = torch.quantile(flat, qs).tolist()

    return ActStats(
        layer=layer_name,
        shape=tuple(x.shape),
        mean=flat.mean().item(),
        std=flat.std(unbiased=False).item(),
        min=flat.min().item(),
        max=flat.max().item(),
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        zero_frac=(flat == 0).float().mean().item(),
    )


class ActivationLogger:
    """
    Hook-based activation logger.

    What it does:
      - registers forward hooks for selected layer types
      - caches the latest activation output per layer
      - can print a stats table
      - can plot one histogram per layer

    Why hooks:
      - you do NOT need to edit model.forward()
      - you can choose exactly which layers to inspect

    Typical usage:
      logger = ActivationLogger(model, layer_types=(nn.Conv2d, nn.ReLU, nn.Linear))
      logger.register()
      logger.clear()
      _ = model(x)
      logger.print_stats()
      logger.plot_histograms()
      logger.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_types: Tuple[type, ...] = (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear),
        include_names: Optional[Iterable[str]] = None,
        exclude_names: Optional[Iterable[str]] = None,
        capture: str = "output",  # "output" is typical for activations
    ):
        self.model = model
        self.layer_types = layer_types
        self.include_names = set(include_names) if include_names is not None else None
        self.exclude_names = set(exclude_names) if exclude_names is not None else set()
        self.capture = capture

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.acts: Dict[str, torch.Tensor] = {}

    def _should_track(self, name: str, module: nn.Module) -> bool:
        """Filter layers by type and optional include/exclude lists."""
        if not isinstance(module, self.layer_types):
            return False
        if self.include_names is not None and name not in self.include_names:
            return False
        if name in self.exclude_names:
            return False
        return True

    def register(self) -> None:
        """Attach forward hooks."""
        self.remove()

        for name, module in self.model.named_modules():
            if name == "":
                continue
            if not self._should_track(name, module):
                continue

            def make_hook(layer_name: str):
                def hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any):
                    # Capture either input or output tensor for this module
                    if self.capture == "input":
                        if len(inputs) == 0 or not torch.is_tensor(inputs[0]):
                            return
                        act = inputs[0]
                    else:
                        if torch.is_tensor(output):
                            act = output
                        elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
                            act = output[0]
                        else:
                            return

                    # Store only the most recent activation for each layer name.
                    self.acts[layer_name] = act.detach()
                return hook

            self.handles.append(module.register_forward_hook(make_hook(name)))

    def clear(self) -> None:
        """Clear cached activations (call before the forward pass you want to inspect)."""
        self.acts.clear()

    def remove(self) -> None:
        """Remove hooks."""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def stats(self) -> Dict[str, ActStats]:
        """Compute stats for all cached activations."""
        return {name: compute_stats(act, name) for name, act in self.acts.items()}

    def print_stats(self, sort_by: str = "layer") -> None:
        """
        Print a compact stats table.

        sort_by can be:
          - "layer" (default)
          - "std", "max", "zero_frac", etc. (any ActStats field)
        """
        rows = list(self.stats().values())

        if sort_by != "layer":
            rows.sort(key=lambda r: getattr(r, sort_by), reverse=True)
        else:
            rows.sort(key=lambda r: r.layer)

        header = "layer | shape | mean | std | min | max | p5 | p50 | p95 | zero"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r.layer:28s} | {str(r.shape):14s} | "
                f"{r.mean: .4f} | {r.std: .4f} | {r.min: .4f} | {r.max: .4f} | "
                f"{r.p5: .4f} | {r.p50: .4f} | {r.p95: .4f} | {r.zero_frac: .3f}"
            )

    def plot_histograms(
        self,
        layer_names: Optional[Iterable[str]] = None,
        bins: int = 60,
        clamp: Optional[Tuple[float, float]] = None,
        max_layers: Optional[int] = None,
    ) -> None:
        """
        Plot activation distributions (histograms).

        Notes:
          - One figure per layer keeps it simple and readable.
          - clamp is useful when outliers make the histogram unreadable.
            Example for ReLU nets: clamp=(0, 10)
        """
        names = list(layer_names) if layer_names is not None else sorted(self.acts.keys())
        if max_layers is not None:
            names = names[:max_layers]

        for name in names:
            act = self.acts.get(name)
            if act is None:
                continue

            flat = _flatten_cpu_float(act)
            if clamp is not None:
                lo, hi = clamp
                flat = flat.clamp(lo, hi)

            plt.figure()
            plt.hist(flat.numpy(), bins=bins)
            plt.title(f"Activation Distribution: {name}  shape={tuple(act.shape)}")
            plt.xlabel("activation value")
            plt.ylabel("frequency")
            plt.show()


# =============================================================================
# 4) OPTIONAL: CONV PER-CHANNEL QUICK CHECK (dead/exploding filters)
# =============================================================================

def plot_conv_per_channel_stats(activation: torch.Tensor, layer_name: str) -> None:
    """
    For conv activations shaped [N, C, H, W], plot per-channel mean/std/zero_frac.

    Why it helps:
      - dead filters => std ~ 0 and/or zero_frac ~ 1 for some channels
      - unstable filters => unusually large std for a few channels

    Usage pattern:
      act = logger.acts["features.0.block.0"]  # example conv layer
      plot_conv_per_channel_stats(act, "features.0.block.0")
    """
    if activation.ndim != 4:
        raise ValueError(f"{layer_name}: expected activation shape [N,C,H,W], got {tuple(activation.shape)}")

    x = activation.detach().float()

    # reduce across batch and spatial dims; keep channels
    channel_mean = x.mean(dim=(0, 2, 3)).cpu()
    channel_std = x.std(dim=(0, 2, 3), unbiased=False).cpu()
    channel_zero = (x == 0).float().mean(dim=(0, 2, 3)).cpu()

    plt.figure()
    plt.plot(channel_std.numpy())
    plt.title(f"Per-Channel Std: {layer_name}")
    plt.xlabel("channel")
    plt.ylabel("std")
    plt.show()

    plt.figure()
    plt.plot(channel_mean.numpy())
    plt.title(f"Per-Channel Mean: {layer_name}")
    plt.xlabel("channel")
    plt.ylabel("mean")
    plt.show()

    plt.figure()
    plt.plot(channel_zero.numpy())
    plt.title(f"Per-Channel Zero Fraction: {layer_name}")
    plt.xlabel("channel")
    plt.ylabel("zero fraction")
    plt.show()
