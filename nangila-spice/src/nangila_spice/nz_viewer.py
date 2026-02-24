"""
Nangila .nz Waveform Viewer

Matplotlib plugin for visualizing .nz compressed waveform files.
Provides interactive plotting with node selection, time zoom,
and error-bound overlay.

Phase 2, Sprint 7 deliverable.
"""

import os
from typing import List, Optional

from .nz_reader import NzWaveform, read_nz


def plot_nz(
    source,
    nodes: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_error_bound: bool = False,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    time_scale: str = "auto",
):
    """Plot waveforms from a .nz file or NzWaveform object.

    Args:
        source: Path to .nz file, or NzWaveform object
        nodes: List of node names to plot (None = all)
        title: Plot title
        show_error_bound: Overlay ±error_bound shading
        output_path: Save to file instead of displaying
        figsize: Figure size (width, height)
        time_scale: "auto", "ps", "ns", "us", or "ms"
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[nz-viewer] matplotlib not installed. Install with: pip install matplotlib")
        return

    # Load data
    if isinstance(source, str):
        wf = read_nz(source)
    elif isinstance(source, NzWaveform):
        wf = source
    else:
        raise TypeError(f"Expected str or NzWaveform, got {type(source)}")

    # Select nodes
    if nodes is None:
        nodes = wf.node_names
    else:
        for n in nodes:
            if n not in wf.signals:
                print(f"[nz-viewer] Warning: node '{n}' not found, skipping")
        nodes = [n for n in nodes if n in wf.signals]

    if not nodes:
        print("[nz-viewer] No nodes to plot")
        return

    # Determine time scale
    time_range = wf.header.t_end - wf.header.t_start
    if time_scale == "auto":
        if time_range < 1e-9:
            time_scale = "ps"
        elif time_range < 1e-6:
            time_scale = "ns"
        elif time_range < 1e-3:
            time_scale = "us"
        else:
            time_scale = "ms"

    scale_factors = {"ps": 1e12, "ns": 1e9, "us": 1e6, "ms": 1e3}
    scale = scale_factors.get(time_scale, 1e9)
    scaled_time = [t * scale for t in wf.time]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    colors = [
        "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
        "#FF9800", "#00BCD4", "#E91E63", "#607D8B",
    ]

    for i, node in enumerate(nodes):
        color = colors[i % len(colors)]
        values = wf.signals[node]
        ax.plot(scaled_time, values, color=color, linewidth=1.5, label=node)

        # Error bound shading
        if show_error_bound and wf.header.error_bound > 0:
            eb = wf.header.error_bound
            upper = [v + eb for v in values]
            lower = [v - eb for v in values]
            ax.fill_between(
                scaled_time, lower, upper,
                alpha=0.1, color=color,
                label=f"±{eb:.1e}V bound",
            )

    # Styling
    ax.set_xlabel(f"Time ({time_scale})", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Title
    if title is None:
        title = "Nangila Waveform"
        if hasattr(wf, "stats") and wf.stats:
            ratio = wf.stats.get("compression_ratio", 0)
            title += f" ({ratio:.1f}× compressed)"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[nz-viewer] Saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def compare_waveforms(
    original_time: List[float],
    original_values: List[float],
    nz_source,
    node: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """Compare original vs .nz-compressed waveform to visualize error.

    Args:
        original_time: Original time values
        original_values: Original voltage values
        nz_source: Path to .nz file or NzWaveform
        node: Node name to compare
        title: Plot title
        output_path: Save to file
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("[nz-viewer] matplotlib not installed")
        return

    if isinstance(nz_source, str):
        wf = read_nz(nz_source)
    else:
        wf = nz_source

    if node not in wf.signals:
        print(f"[nz-viewer] Node '{node}' not in .nz file")
        return

    compressed = wf.signals[node]

    # Time scaling
    time_range = max(original_time) - min(original_time)
    if time_range < 1e-9:
        scale, unit = 1e12, "ps"
    elif time_range < 1e-6:
        scale, unit = 1e9, "ns"
    else:
        scale, unit = 1e6, "μs"

    t_scaled = [t * scale for t in original_time]
    t_nz = [t * scale for t in wf.time]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # Top: overlay
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_scaled, original_values, "b-", linewidth=1.5, label="Original")
    ax1.plot(t_nz, compressed, "r--", linewidth=1.0, alpha=0.8, label=".nz Compressed")
    ax1.set_ylabel("Voltage (V)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or f"Waveform Comparison: {node}", fontsize=14, fontweight="bold")

    # Bottom: error
    ax2 = fig.add_subplot(gs[1])
    n = min(len(original_values), len(compressed))
    errors = [abs(original_values[i] - compressed[i]) for i in range(n)]
    t_err = t_scaled[:n]
    ax2.semilogy(t_err, errors, "k-", linewidth=0.8)
    ax2.axhline(y=wf.header.error_bound, color="r", linestyle="--", alpha=0.7,
                label=f"Error bound: {wf.header.error_bound:.1e}V")
    ax2.set_xlabel(f"Time ({unit})", fontsize=12)
    ax2.set_ylabel("|Error| (V)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[nz-viewer] Saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()
