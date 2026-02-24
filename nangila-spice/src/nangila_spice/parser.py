"""
SPICE Netlist Parser

Parses standard SPICE netlist files (.sp, .cir) into an internal
circuit graph representation suitable for partitioning.

Phase 1, Sprint 1 deliverable.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Device:
    """A single circuit element (resistor, capacitor, MOSFET, etc.)."""
    name: str
    dev_type: str  # 'R', 'C', 'L', 'M', 'V', 'I', 'X' (subckt)
    nodes: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
    model: Optional[str] = None


@dataclass
class Subcircuit:
    """A subcircuit definition (.subckt ... .ends)."""
    name: str
    ports: list[str] = field(default_factory=list)
    devices: list[Device] = field(default_factory=list)


@dataclass
class Netlist:
    """Parsed representation of a SPICE netlist."""
    title: str = ""
    devices: list[Device] = field(default_factory=list)
    subcircuits: dict[str, Subcircuit] = field(default_factory=dict)
    global_nodes: set[str] = field(default_factory=set)

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def num_nodes(self) -> int:
        nodes = set()
        for dev in self.devices:
            nodes.update(dev.nodes)
        return len(nodes)

    def flatten(self) -> "Netlist":
        """
        Produce a flattened version of this netlist where all subcircuits (X)
        are expanded into their primitive devices.
        """
        flat = Netlist(title=self.title, global_nodes=self.global_nodes)
        
        for device in self.devices:
            if device.dev_type == "X":
                self._flatten_recursive(device.model, device.name, device.nodes, "", flat.devices)
            else:
                flat.devices.append(device)
                
        return flat

    def _flatten_recursive(self, subckt_name: str, instance_name: str, actual_nodes: list[str], prefix: str, target_devices: list[Device]):
        """Recursively expand a subcircuit instance into flat devices."""
        subckt = self.subcircuits.get(subckt_name)
        if not subckt:
            return

        port_map = dict(zip(subckt.ports, actual_nodes))
        new_prefix = f"{prefix}{instance_name}."
        
        for dev in subckt.devices:
            # Map nodes (ports → actual nodes, others → prefixed internal nodes)
            mapped_nodes = []
            for node in dev.nodes:
                if node in port_map:
                    mapped_nodes.append(port_map[node])
                elif node in ("0", "gnd") or node in self.global_nodes:
                    mapped_nodes.append(node)
                else:
                    mapped_nodes.append(f"{new_prefix}{node}")
            
            if dev.dev_type == "X":
                self._flatten_recursive(dev.model, dev.name, mapped_nodes, new_prefix, target_devices)
            else:
                # Create a new primitive device with mapped nodes and prefixed name.
                # We preserve the first character (type) to keep SPICE compatibility.
                # Example: R1 in subckt X1 becomes RX1.R1
                safe_prefix = new_prefix.replace(".", "_")
                new_dev_name = f"{dev.name[0]}{safe_prefix}{dev.name[1:]}"
                
                new_dev = Device(
                    name=new_dev_name,
                    dev_type=dev.dev_type,
                    nodes=mapped_nodes,
                    params=dev.params.copy(),
                    model=dev.model,
                )
                target_devices.append(new_dev)


def parse_netlist(filepath: str) -> Netlist:
    """
    Parse a SPICE netlist file into our internal representation.

    Args:
        filepath: Path to .sp or .cir file.

    Returns:
        Parsed Netlist object.
    """
    netlist = Netlist()

    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        return netlist

    # First line is always the title
    netlist.title = lines[0].strip()

    current_subckt: Optional[Subcircuit] = None
    i = 1

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith("*"):
            i += 1
            continue

        # Handle continuation lines (start with +)
        while i + 1 < len(lines) and lines[i + 1].strip().startswith("+"):
            i += 1
            line += " " + lines[i].strip()[1:]

        upper = line.upper()

        # Control statements
        if upper.startswith(".SUBCKT"):
            parts = line.split()
            subckt = Subcircuit(name=parts[1], ports=parts[2:])
            current_subckt = subckt
        elif upper.startswith(".ENDS"):
            if current_subckt:
                netlist.subcircuits[current_subckt.name] = current_subckt
                current_subckt = None
        elif upper.startswith("."):
            # Skip other control statements for now
            pass
        else:
            # Device instance
            device = _parse_device_line(line)
            if device:
                if current_subckt:
                    current_subckt.devices.append(device)
                else:
                    netlist.devices.append(device)

        i += 1

    return netlist


def _parse_device_line(line: str) -> Optional[Device]:
    """Parse a single device instance line."""
    parts = line.split()
    if not parts:
        return None

    name = parts[0]
    dev_type = name[0].upper()

    if dev_type in ("R", "C", "L"):
        # Two-terminal: R1 n1 n2 value
        if len(parts) >= 4:
            return Device(
                name=name,
                dev_type=dev_type,
                nodes=[parts[1], parts[2]],
                params={"value": parts[3]},
            )
    elif dev_type == "M":
        # MOSFET: M1 drain gate source bulk model [params]
        if len(parts) >= 6:
            params = {}
            for p in parts[6:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    params[k] = v
            return Device(
                name=name,
                dev_type=dev_type,
                nodes=[parts[1], parts[2], parts[3], parts[4]],
                model=parts[5],
                params=params,
            )
    elif dev_type in ("V", "I"):
        # Source: V1 n+ n- value
        if len(parts) >= 4:
            return Device(
                name=name,
                dev_type=dev_type,
                nodes=[parts[1], parts[2]],
                params={"value": " ".join(parts[3:])},
            )
    elif dev_type == "X":
        # Subcircuit instance: X1 n1 n2 ... subckt_name
        if len(parts) >= 3:
            return Device(
                name=name,
                dev_type=dev_type,
                nodes=parts[1:-1],
                model=parts[-1],  # subcircuit name
            )

    # Fallback: generic parse
    return Device(name=name, dev_type=dev_type, nodes=parts[1:])
