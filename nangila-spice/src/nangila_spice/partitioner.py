"""
Circuit Graph Partitioner

Converts a parsed netlist into a hypergraph and partitions it using
METIS (via pymetis) or a built-in spectral method for distributed
simulation across multiple solver nodes.

Supports:
  - METIS-based balanced k-way partitioning
  - Spectral bisection fallback (no external deps)
  - Analog feedback loop constraints (devices kept together)
  - Boundary (Ghost) node identification

Phase 1, Sprint 2 deliverable.
"""

from dataclasses import dataclass, field
from typing import Optional
import warnings

from .parser import Netlist
from .graph import CircuitGraph, build_circuit_graph

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")


# --- Data structures ---

@dataclass
class Partition:
    """A single partition of the circuit graph."""
    partition_id: int
    device_names: list[str] = field(default_factory=list)
    internal_nodes: set[str] = field(default_factory=set)
    boundary_nodes: set[str] = field(default_factory=set)  # Ghost nodes
    num_devices: int = 0

    @property
    def ghost_count(self) -> int:
        return len(self.boundary_nodes)


@dataclass
class PartitionResult:
    """Result of partitioning a netlist."""
    partitions: list[Partition] = field(default_factory=list)
    total_cut_edges: int = 0
    total_boundary_nodes: int = 0
    method: str = "unknown"
    feedback_groups_enforced: int = 0
    balance_ratio: float = 0.0  # max_size / avg_size

    def summary(self) -> str:
        """Human-readable partition summary."""
        sizes = [p.num_devices for p in self.partitions]
        lines = [
            f"Partitioner: {self.method}",
            f"  Partitions: {len(self.partitions)}",
            f"  Devices:    {sum(sizes)} (min={min(sizes)}, max={max(sizes)}, "
            f"avg={sum(sizes)/len(sizes):.1f})",
            f"  Balance:    {self.balance_ratio:.2f}x",
            f"  Boundary:   {self.total_boundary_nodes} ghost nodes",
            f"  Cut edges:  {self.total_cut_edges}",
            f"  Constraints: {self.feedback_groups_enforced} feedback groups enforced",
        ]
        return "\n".join(lines)


# --- Main entry point ---

def partition_netlist(
    netlist: Netlist,
    k: int,
    method: str = "auto",
    imbalance_tolerance: float = 1.05,
) -> PartitionResult:
    """
    Partition a netlist into k balanced sub-graphs.

    Args:
        netlist: Parsed netlist.
        k: Number of partitions (should match available cores).
        method: 'metis', 'spectral', or 'auto' (try metis, fallback to spectral).
        imbalance_tolerance: Max allowed ratio of largest to average partition (1.0 = perfect).

    Returns:
        PartitionResult with k partitions and identified boundary nodes.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k == 1:
        return _single_partition(netlist)

    # Build circuit graph
    cg = build_circuit_graph(netlist)

    # Choose partitioning method
    if method == "auto":
        try:
            import pymetis  # noqa: F401
            method = "metis"
        except ImportError:
            method = "spectral"

    if method == "metis":
        device_to_part = _partition_metis(cg, k, imbalance_tolerance)
    elif method == "spectral":
        device_to_part = _partition_spectral(cg, k)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'metis', 'spectral', or 'auto'.")

    # Enforce feedback constraints (move coupled devices to same partition)
    constraints_enforced = _enforce_feedback_constraints(
        device_to_part, cg.feedback_groups
    )

    # Build result
    result = _build_result(netlist, cg, device_to_part, method, constraints_enforced)

    return result


# --- Partitioning algorithms ---

def _partition_metis(
    cg: CircuitGraph, k: int, imbalance_tol: float
) -> dict[str, int]:
    """Partition using METIS (requires pymetis)."""
    import pymetis

    # Build adjacency list for METIS (device nodes only)
    device_list = sorted(cg.device_nodes)
    dev_to_idx = {d: i for i, d in enumerate(device_list)}
    n = len(device_list)

    # Build adjacency through shared nets
    adjacency: list[list[int]] = [[] for _ in range(n)]
    edge_weights_list: list[list[int]] = [[] for _ in range(n)]

    # For each net, connect all devices sharing that net
    net_to_devices: dict[str, list[str]] = {}
    for dev_id in device_list:
        for neighbor in cg.graph.neighbors(dev_id):
            if cg.graph.nodes[neighbor].get("ntype") == "net":
                if neighbor not in net_to_devices:
                    net_to_devices[neighbor] = []
                net_to_devices[neighbor].append(dev_id)

    # Skip supply nets
    supply_prefixes = {"N:0", "N:gnd", "N:vdd", "N:vss", "N:vcc"}
    for net_id, devs in net_to_devices.items():
        if net_id.lower() in supply_prefixes:
            continue
        if len(devs) > 50:  # High-fanout net
            continue
        for i_d in range(len(devs)):
            for j_d in range(i_d + 1, len(devs)):
                d1_idx = dev_to_idx[devs[i_d]]
                d2_idx = dev_to_idx[devs[j_d]]
                # Get max weight of connections through this net
                w1 = cg.graph.edges.get((devs[i_d], net_id), {}).get("weight", 1)
                w2 = cg.graph.edges.get((devs[j_d], net_id), {}).get("weight", 1)
                w = max(w1, w2)
                adjacency[d1_idx].append(d2_idx)
                adjacency[d2_idx].append(d1_idx)
                edge_weights_list[d1_idx].append(w)
                edge_weights_list[d2_idx].append(w)

    # Call METIS
    try:
        _cut_count, membership = pymetis.part_graph(
            k,
            adjacency=adjacency,
            # Note: pymetis edge weights via xadj/adjncy format
        )
    except Exception as e:
        warnings.warn(f"METIS failed ({e}), falling back to spectral")
        return _partition_spectral(cg, k)

    # Map back to device names
    device_to_part: dict[str, int] = {}
    for idx, dev_id in enumerate(device_list):
        # Strip 'D:' prefix to get original device name
        dev_name = dev_id[2:]
        device_to_part[dev_name] = membership[idx]

    return device_to_part


def _partition_spectral(cg: CircuitGraph, k: int) -> dict[str, int]:
    """
    Spectral bisection partitioning (no external dependencies beyond NetworkX).

    Uses the Fiedler vector (second smallest eigenvector of the Laplacian)
    to recursively bisect the graph.
    """
    device_list = sorted(cg.device_nodes)

    if len(device_list) <= k:
        return {d[2:]: i for i, d in enumerate(device_list)}

    # Build device-device adjacency subgraph via shared nets
    dev_graph = nx.Graph()
    for dev_id in device_list:
        dev_graph.add_node(dev_id)

    net_to_devices: dict[str, list[str]] = {}
    for dev_id in device_list:
        for neighbor in cg.graph.neighbors(dev_id):
            if cg.graph.nodes[neighbor].get("ntype") == "net":
                if neighbor not in net_to_devices:
                    net_to_devices[neighbor] = []
                net_to_devices[neighbor].append(dev_id)

    supply_prefixes = {"N:0", "N:gnd", "N:vdd", "N:vss", "N:vcc"}
    for net_id, devs in net_to_devices.items():
        if net_id.lower() in supply_prefixes:
            continue
        for i_d in range(len(devs)):
            for j_d in range(i_d + 1, len(devs)):
                if dev_graph.has_edge(devs[i_d], devs[j_d]):
                    dev_graph[devs[i_d]][devs[j_d]]["weight"] += 1
                else:
                    dev_graph.add_edge(devs[i_d], devs[j_d], weight=1)

    # Recursive bisection
    partition_map: dict[str, int] = {}
    _recursive_bisect(dev_graph, list(dev_graph.nodes), k, 0, partition_map)

    # Strip D: prefix
    return {d[2:]: p for d, p in partition_map.items()}


def _recursive_bisect(
    graph: nx.Graph,
    nodes: list[str],
    k: int,
    offset: int,
    partition_map: dict[str, int],
) -> None:
    """Recursively bisect a node list into k partitions."""
    if k <= 1 or len(nodes) <= 1:
        for n in nodes:
            partition_map[n] = offset
        return

    subgraph = graph.subgraph(nodes)

    # Try spectral bisection via Fiedler vector
    try:
        if nx.is_connected(subgraph):
            fiedler = nx.fiedler_vector(subgraph, weight="weight")
        else:
            # Handle disconnected: partition by components first
            components = list(nx.connected_components(subgraph))
            mid = len(nodes) // 2
            left, right = [], []
            count = 0
            for comp in sorted(components, key=len, reverse=True):
                target = left if count < mid else right
                target.extend(comp)
                count += len(comp)
            if not left:
                left = right[:len(right) // 2]
                right = right[len(right) // 2:]

            k_left = max(1, k // 2)
            k_right = k - k_left
            _recursive_bisect(graph, left, k_left, offset, partition_map)
            _recursive_bisect(graph, right, k_right, offset + k_left, partition_map)
            return
    except Exception:
        # Fallback: simple split by index
        mid = len(nodes) // 2
        fiedler = None

    node_list = list(subgraph.nodes)
    if fiedler is not None:
        # Sort by Fiedler value and split
        paired = sorted(zip(fiedler, node_list), key=lambda x: x[0])
        sorted_nodes = [n for _, n in paired]
    else:
        sorted_nodes = node_list

    mid = len(sorted_nodes) // 2
    left = sorted_nodes[:mid]
    right = sorted_nodes[mid:]

    k_left = max(1, k // 2)
    k_right = k - k_left
    _recursive_bisect(graph, left, k_left, offset, partition_map)
    _recursive_bisect(graph, right, k_right, offset + k_left, partition_map)


# --- Constraint enforcement ---

def _enforce_feedback_constraints(
    device_to_part: dict[str, int],
    feedback_groups: list[set[str]],
) -> int:
    """
    Move all devices in a feedback group to the same partition.
    Uses majority vote: devices move to the partition that holds
    the most group members.

    Returns:
        Number of feedback groups that required enforcement.
    """
    enforced = 0
    for group in feedback_groups:
        # Get current partition assignments
        parts = {}
        for dev in group:
            if dev in device_to_part:
                p = device_to_part[dev]
                parts[p] = parts.get(p, 0) + 1

        if len(parts) <= 1:
            continue  # Already in same partition

        # Move all to dominant partition (majority vote)
        dominant = max(parts, key=parts.get)
        for dev in group:
            if dev in device_to_part:
                device_to_part[dev] = dominant
        enforced += 1

    return enforced


# --- Result building ---

def _build_result(
    netlist: Netlist,
    cg: CircuitGraph,
    device_to_part: dict[str, int],
    method: str,
    constraints_enforced: int,
) -> PartitionResult:
    """Build the final PartitionResult from device assignments."""
    # Find all partition IDs
    all_parts = set(device_to_part.values())
    k = max(all_parts) + 1 if all_parts else 1

    partitions = [Partition(partition_id=i) for i in range(k)]

    # Assign devices to partitions
    for device in netlist.devices:
        p_id = device_to_part.get(device.name, 0)
        partitions[p_id].device_names.append(device.name)
        partitions[p_id].internal_nodes.update(device.nodes)
        partitions[p_id].num_devices += 1

    # Identify boundary nodes (shared between partitions)
    node_partitions: dict[str, set[int]] = {}
    for p in partitions:
        for node in p.internal_nodes:
            if node not in node_partitions:
                node_partitions[node] = set()
            node_partitions[node].add(p.partition_id)

    total_cut = 0
    for node, pset in node_partitions.items():
        if len(pset) > 1:
            total_cut += 1
            for p_id in pset:
                partitions[p_id].boundary_nodes.add(node)

    # Remove boundary from internal
    for p in partitions:
        p.internal_nodes -= p.boundary_nodes

    # Compute balance
    sizes = [p.num_devices for p in partitions if p.num_devices > 0]
    avg_size = sum(sizes) / len(sizes) if sizes else 1
    max_size = max(sizes) if sizes else 1
    balance = max_size / avg_size if avg_size > 0 else 1.0

    result = PartitionResult(
        partitions=partitions,
        total_cut_edges=total_cut,
        total_boundary_nodes=len(
            set().union(*(p.boundary_nodes for p in partitions)) if partitions else set()
        ),
        method=method,
        feedback_groups_enforced=constraints_enforced,
        balance_ratio=balance,
    )

    return result


def _single_partition(netlist: Netlist) -> PartitionResult:
    """Trivial case: everything in one partition."""
    p = Partition(partition_id=0)
    for device in netlist.devices:
        p.device_names.append(device.name)
        p.internal_nodes.update(device.nodes)
        p.num_devices += 1

    return PartitionResult(
        partitions=[p],
        total_cut_edges=0,
        total_boundary_nodes=0,
        method="single",
        balance_ratio=1.0,
    )
