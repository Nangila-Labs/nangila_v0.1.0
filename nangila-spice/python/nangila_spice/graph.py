"""
Circuit Graph Builder

Converts a parsed Netlist into a NetworkX graph suitable for
partitioning. Builds a hypergraph where:
  - Each device is a hyperedge connecting its terminal nodes
  - Each net (circuit node) is a graph vertex

Also detects analog feedback loops that must be kept in the
same partition to preserve convergence.

Phase 1, Sprint 2 deliverable.
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "NetworkX is required for graph partitioning. "
        "Install with: pip install networkx"
    )

from .parser import Netlist, Device


@dataclass
class CircuitGraph:
    """A graph representation of a circuit for partitioning.

    Uses a "star expansion" of the hypergraph:
      - Each circuit net → graph node (type='net')
      - Each device → graph node (type='device')
      - Edges connect devices to their terminal nets

    This preserves the hypergraph structure while allowing
    standard graph partitioning algorithms.
    """
    graph: nx.Graph = field(default_factory=nx.Graph)
    net_nodes: set[str] = field(default_factory=set)
    device_nodes: set[str] = field(default_factory=set)
    feedback_groups: list[set[str]] = field(default_factory=list)

    @property
    def num_nets(self) -> int:
        return len(self.net_nodes)

    @property
    def num_devices(self) -> int:
        return len(self.device_nodes)

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()


def build_circuit_graph(netlist: Netlist) -> CircuitGraph:
    """
    Convert a parsed netlist into a circuit graph.

    Each device becomes a node connected to its terminal net nodes.
    Device nodes are prefixed with 'D:' and net nodes with 'N:' to
    avoid name collisions.

    Args:
        netlist: Parsed Netlist from parser.py.

    Returns:
        CircuitGraph with device/net nodes and edges.
    """
    cg = CircuitGraph()

    for device in netlist.devices:
        dev_id = f"D:{device.name}"
        cg.graph.add_node(dev_id, ntype="device", dev_type=device.dev_type)
        cg.device_nodes.add(dev_id)

        for net_name in device.nodes:
            net_id = f"N:{net_name}"
            if net_id not in cg.net_nodes:
                cg.graph.add_node(net_id, ntype="net")
                cg.net_nodes.add(net_id)

            # Edge weight: higher for signal-critical connections
            weight = _connection_weight(device, net_name)
            cg.graph.add_edge(dev_id, net_id, weight=weight)

    # Detect feedback loops
    cg.feedback_groups = detect_feedback_loops(cg, netlist)

    return cg


def _connection_weight(device: Device, net_name: str) -> int:
    """
    Assign edge weight based on connection criticality.

    Higher weights discourage the partitioner from cutting these edges,
    keeping tightly-coupled devices together.
    """
    # Gate connections of MOSFETs are highly sensitive
    if device.dev_type == "M" and len(device.nodes) >= 2:
        if net_name == device.nodes[1]:  # gate terminal
            return 10

    # Supply/ground nets should be freely shared (low weight)
    if net_name.lower() in ("0", "gnd", "vdd", "vss", "vcc"):
        return 1

    # Default
    return 5


def detect_feedback_loops(
    cg: CircuitGraph, netlist: Netlist
) -> list[set[str]]:
    """
    Detect analog feedback loops that must stay in the same partition.

    Strategy:
      1. Find cross-coupled pairs (e.g., SRAM latch)
      2. Find short cycles in the device connectivity graph

    Returns:
        List of device-name sets, each set must stay in one partition.
    """
    feedback_groups: list[set[str]] = []

    # --- Pattern 1: Cross-coupled pairs ---
    # Two devices whose outputs drive each other's inputs.
    # Common in: SRAM cells, latches, sense amps, oscillators.
    mosfets = [d for d in netlist.devices if d.dev_type == "M"]

    for i, m1 in enumerate(mosfets):
        for m2 in mosfets[i + 1:]:
            if len(m1.nodes) >= 4 and len(m2.nodes) >= 4:
                # Check if M1's drain connects to M2's gate and vice versa
                m1_drain, m1_gate = m1.nodes[0], m1.nodes[1]
                m2_drain, m2_gate = m2.nodes[0], m2.nodes[1]

                if m1_drain == m2_gate and m2_drain == m1_gate:
                    group = {m1.name, m2.name}
                    feedback_groups.append(group)

    # --- Pattern 2: Short cycles (length ≤ 4 devices) ---
    # Build a device-only adjacency graph
    device_adj: dict[str, set[str]] = {}
    net_to_devices: dict[str, list[str]] = {}

    for device in netlist.devices:
        device_adj[device.name] = set()
        for net in device.nodes:
            if net not in net_to_devices:
                net_to_devices[net] = []
            net_to_devices[net].append(device.name)

    # Skip supply nets (connected to too many devices)
    supply_nets = {"0", "gnd", "vdd", "vss", "vcc"}
    for net, devs in net_to_devices.items():
        if net.lower() in supply_nets:
            continue
        if len(devs) > 20:  # High-fanout net, skip
            continue
        for d1 in devs:
            for d2 in devs:
                if d1 != d2:
                    device_adj[d1].add(d2)

    # Find 3-cycles (tight feedback)
    visited_triples: set[tuple[str, ...]] = set()
    for d1, neighbors in device_adj.items():
        for d2 in neighbors:
            for d3 in device_adj.get(d2, set()):
                if d3 != d1 and d1 in device_adj.get(d3, set()):
                    triple = tuple(sorted([d1, d2, d3]))
                    if triple not in visited_triples:
                        visited_triples.add(triple)
                        feedback_groups.append({d1, d2, d3})

    # Merge overlapping groups
    feedback_groups = _merge_overlapping_groups(feedback_groups)

    return feedback_groups


def _merge_overlapping_groups(groups: list[set[str]]) -> list[set[str]]:
    """Merge groups that share any device."""
    if not groups:
        return []

    merged = True
    while merged:
        merged = False
        new_groups: list[set[str]] = []
        used = [False] * len(groups)

        for i in range(len(groups)):
            if used[i]:
                continue
            current = set(groups[i])
            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                if current & groups[j]:  # overlap
                    current |= groups[j]
                    used[j] = True
                    merged = True
            new_groups.append(current)
            used[i] = True

        groups = new_groups

    return groups
