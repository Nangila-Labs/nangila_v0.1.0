"""
Partition Visualization

Renders partitioned circuit graphs for debugging and analysis.
Uses matplotlib for static 2D graph layouts with partition coloring.

Phase 1, Sprint 2 deliverable.
"""

from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")

from .graph import CircuitGraph
from .partitioner import PartitionResult


# Color palette for partitions (up to 16 distinct colors)
PARTITION_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Sky blue
    "#96CEB4",  # Sage
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light blue
    "#F0B27A",  # Peach
    "#82E0AA",  # Green
    "#F1948A",  # Pink
    "#AED6F1",  # Powder blue
    "#D7BDE2",  # Lavender
    "#A9DFBF",  # Pale green
]


def visualize_partitions(
    cg: CircuitGraph,
    result: PartitionResult,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_nets: bool = False,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Render a partitioned circuit graph.

    Args:
        cg: The circuit graph.
        result: Partitioning result.
        output_path: Save to file if provided (e.g., 'partition.png').
        title: Plot title.
        show_nets: If True, also show net nodes (can be noisy).
        figsize: Figure size in inches.
    """
    if not HAS_MATPLOTLIB:
        print("[viz] matplotlib not installed, skipping visualization.")
        print("[viz] Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Build device-to-partition mapping
    dev_to_part: dict[str, int] = {}
    for p in result.partitions:
        for dev_name in p.device_names:
            dev_to_part[dev_name] = p.partition_id

    # Filter nodes to show
    if show_nets:
        subgraph = cg.graph
    else:
        device_only = [n for n in cg.graph.nodes if n.startswith("D:")]
        subgraph = cg.graph.subgraph(device_only)

    if len(subgraph) == 0:
        print("[viz] No nodes to display.")
        return

    # Layout
    try:
        pos = nx.spring_layout(subgraph, seed=42, k=2.0, iterations=50)
    except Exception:
        pos = nx.circular_layout(subgraph)

    # Draw each partition in a different color
    k = len(result.partitions)
    for p_id in range(k):
        color = PARTITION_COLORS[p_id % len(PARTITION_COLORS)]
        dev_names_in_part = set(result.partitions[p_id].device_names)
        nodes_in_part = [
            n for n in subgraph.nodes
            if n.startswith("D:") and n[2:] in dev_names_in_part
        ]

        if nodes_in_part:
            nx.draw_networkx_nodes(
                subgraph, pos,
                nodelist=nodes_in_part,
                node_color=color,
                node_size=80,
                alpha=0.85,
                ax=ax,
            )

    # Draw net nodes if shown
    if show_nets:
        net_nodes = [n for n in subgraph.nodes if n.startswith("N:")]
        if net_nodes:
            nx.draw_networkx_nodes(
                subgraph, pos,
                nodelist=net_nodes,
                node_color="#CCCCCC",
                node_size=20,
                alpha=0.5,
                ax=ax,
                node_shape="s",
            )

    # Draw edges
    # Highlight boundary edges (between partitions) in red
    boundary_edges = []
    internal_edges = []
    for u, v in subgraph.edges():
        u_name = u[2:] if u.startswith("D:") else u
        v_name = v[2:] if v.startswith("D:") else v
        u_part = dev_to_part.get(u_name, -1)
        v_part = dev_to_part.get(v_name, -1)
        if u_part != v_part and u_part >= 0 and v_part >= 0:
            boundary_edges.append((u, v))
        else:
            internal_edges.append((u, v))

    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=internal_edges,
        edge_color="#CCCCCC",
        alpha=0.3,
        width=0.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=boundary_edges,
        edge_color="#FF0000",
        alpha=0.6,
        width=1.0,
        style="dashed",
        ax=ax,
    )

    # Legend
    legend_patches = []
    for p_id in range(min(k, len(PARTITION_COLORS))):
        p = result.partitions[p_id]
        color = PARTITION_COLORS[p_id % len(PARTITION_COLORS)]
        legend_patches.append(
            mpatches.Patch(
                color=color,
                label=f"P{p_id}: {p.num_devices} devs, {p.ghost_count} ghosts"
            )
        )
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

    # Title
    plot_title = title or "Circuit Partition Layout"
    ax.set_title(
        f"{plot_title}\n"
        f"{result.method} | {sum(p.num_devices for p in result.partitions)} devices | "
        f"{result.total_boundary_nodes} ghost nodes | "
        f"balance={result.balance_ratio:.2f}x",
        fontsize=10,
    )
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved partition visualization to {output_path}")
    else:
        plt.show()

    plt.close(fig)
