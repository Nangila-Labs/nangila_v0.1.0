use crate::device_model::{DiodeModel, MosfetLevel1};
use crate::mna::Element;
use crate::ngspice_ffi::PartitionNetlist;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tracing::{debug, error, info};

/// Parses standard SPICE engineering suffixes into f64 multipliers.
/// Handles: f (1e-15), p (1e-12), n (1e-9), u (1e-6), m (1e-3),
///          k (1e3), meg (1e6), g (1e9), t (1e12).
pub fn parse_scale_factor(val: &str) -> Option<f64> {
    let val = val.to_lowercase();
    let split_idx = val.find(|c: char| !c.is_numeric() && c != '.' && c != '+' && c != '-');

    if let Some(idx) = split_idx {
        let (num_str, suffix) = val.split_at(idx);
        let num: f64 = num_str.parse().ok()?;

        let multiplier = match suffix {
            "f" => 1e-15,
            "p" => 1e-12,
            "n" => 1e-9,
            "u" => 1e-6,
            "m" => 1e-3,
            "k" => 1e3,
            "meg" => 1e6,
            "g" => 1e9,
            "t" => 1e12,
            _ => 1.0, // Ignore unknown/trailing characters per SPICE standard
        };
        Some(num * multiplier)
    } else {
        val.parse().ok()
    }
}

/// A parsed subcircuit definition containing its ports and internal lines exactly as text.
#[derive(Debug, Clone)]
struct SubcircuitDef {
    name: String,
    ports: Vec<String>,
    lines: Vec<String>,
}

/// A model card declaration.  We store just the name and type so the Newton-Raphson
/// device evaluator can look up model parameters during stamping.
#[derive(Debug, Clone)]
pub struct ModelCard {
    pub name: String,
    pub model_type: String, // "NMOS", "PMOS", "D", etc.
    pub params: HashMap<String, f64>,
}

/// SPICE Streaming Parser
pub struct SpiceParser {
    node_map: HashMap<String, usize>,
    next_node_id: usize,
    elements: Vec<Element>,
    subcircuits: HashMap<String, SubcircuitDef>,
    /// Key-value parameters defined by `.PARAM name=value`
    params: HashMap<String, f64>,
    /// Model cards defined by `.MODEL name type ...`
    pub models: HashMap<String, ModelCard>,
    /// Mapping: ghost_net_id → (local_node_index, owner_partition_id)
    pub ghost_map: Vec<(u64, usize, u32)>,
}

impl SpiceParser {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        // GND is strictly Node 0
        map.insert("0".to_string(), 0);
        map.insert("gnd".to_string(), 0);

        Self {
            node_map: map,
            next_node_id: 1,
            elements: Vec::new(),
            subcircuits: HashMap::new(),
            params: HashMap::new(),
            models: HashMap::new(),
            ghost_map: Vec::new(),
        }
    }

    /// Resolve a `.PARAM` or engineering-suffix value string, substituting known parameters.
    fn parse_value(&self, s: &str) -> f64 {
        let s = s.trim_matches(|c| c == '\'' || c == '{' || c == '}');
        // Check if it's a named parameter
        if let Some(&v) = self.params.get(&s.to_lowercase()) {
            return v;
        }
        parse_scale_factor(s).unwrap_or(0.0)
    }

    /// Map an alphanumeric SPICE node name to our continuous `usize` integer ID space.
    fn get_or_create_node(&mut self, name: &str) -> usize {
        let name = name.to_lowercase();
        *self.node_map.entry(name).or_insert_with(|| {
            let id = self.next_node_id;
            self.next_node_id += 1;
            id
        })
    }

    /// Iterator that handles SPICE `+` line continuations across a file buffer.
    fn line_continuation_iter<R: BufRead>(reader: R) -> impl Iterator<Item = String> {
        let mut lines = reader.lines().map_while(Result::ok);
        let mut current_stmt = String::new();

        std::iter::from_fn(move || {
            loop {
                let next_line = match lines.next() {
                    Some(l) => l,
                    None => {
                        return if current_stmt.is_empty() {
                            None
                        } else {
                            let res = current_stmt.clone();
                            current_stmt.clear();
                            Some(res)
                        };
                    }
                };

                let trimmed = next_line.trim();
                if trimmed.is_empty() || trimmed.starts_with('*') {
                    continue; // Skip comments and empty lines
                }

                if trimmed.starts_with('+') {
                    // Line continuation: replace '+' with space and append
                    current_stmt.push(' ');
                    current_stmt.push_str(trimmed[1..].trim());
                } else if !current_stmt.is_empty() {
                    // A new statement is beginning. Yield the buffered statement.
                    let yielded = current_stmt.clone();
                    current_stmt = trimmed.to_string();
                    return Some(yielded);
                } else {
                    // First valid line of a new statement
                    current_stmt = trimmed.to_string();
                }
            }
        })
    }

    /// Pass 1: Scan for `.SUBCKT` blocks **and** global `.PARAM` declarations.
    fn discover_subcircuits<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut in_subckt = false;
        let mut current_subckt: Option<SubcircuitDef> = None;

        for line in Self::line_continuation_iter(reader) {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let first = tokens[0].to_uppercase();

            if first == ".SUBCKT" {
                if tokens.len() < 2 {
                    error!("Invalid .SUBCKT declaration: {}", line);
                    continue;
                }
                in_subckt = true;
                let name = tokens[1].to_uppercase();
                let ports: Vec<String> = tokens[2..].iter().map(|s| s.to_string()).collect();
                
                current_subckt = Some(SubcircuitDef {
                    name,
                    ports,
                    lines: Vec::new(),
                });
            } else if first == ".ENDS" {
                in_subckt = false;
                if let Some(def) = current_subckt.take() {
                    debug!("Discovered SUBCKT: {} with {} ports", def.name, def.ports.len());
                    self.subcircuits.insert(def.name.clone(), def);
                }
            } else if first == ".PARAM" && !in_subckt {
                // .PARAM name=value  or  .PARAM name value
                for tok in &tokens[1..] {
                    if let Some(eq_pos) = tok.find('=') {
                        let k = tok[..eq_pos].to_lowercase();
                        let v = parse_scale_factor(&tok[eq_pos+1..]).unwrap_or(0.0);
                        self.params.insert(k, v);
                    }
                }
            } else if first == ".MODEL" && !in_subckt {
                // .MODEL name type [param=value ...]
                if tokens.len() >= 3 {
                    let mname = tokens[1].to_uppercase();
                    let mtype = tokens[2].to_uppercase();
                    let mut mparams: HashMap<String, f64> = HashMap::new();
                    for tok in &tokens[3..] {
                        if let Some(eq_pos) = tok.find('=') {
                            let k = tok[..eq_pos].to_lowercase();
                            let v = parse_scale_factor(&tok[eq_pos+1..]).unwrap_or(0.0);
                            mparams.insert(k, v);
                        }
                    }
                    self.models.insert(mname.clone(), ModelCard { name: mname, model_type: mtype, params: mparams });
                }
            } else if in_subckt {
                if let Some(def) = &mut current_subckt {
                    def.lines.push(line);
                }
            }
        }
        Ok(())
    }

    /// Pass 2: Parse standard elements and recursively flatten `.SUBCKT` instantiations (`X`).
    fn flatten_netlist<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut in_subckt = false;

        for line in Self::line_continuation_iter(reader) {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let first = tokens[0].to_uppercase();

            // Skip lines that are inside a .SUBCKT block definition (we already mapped them in Pass 1)
            if first == ".SUBCKT" {
                in_subckt = true;
                continue;
            }
            if first == ".ENDS" {
                in_subckt = false;
                continue;
            }
            if in_subckt {
                continue;
            }

            // --- NOT inside a subcircuit definition right now ---
            self.parse_element_line(&tokens, "");
        }
        
        Ok(())
    }

    /// Parse a single logic line into an Element, supporting Subcircuit recursive expansion.
    /// `prefix` is prepended to subcircuit internal nodes (e.g., "X1.internal_node").
    fn parse_element_line(&mut self, tokens: &[&str], prefix: &str) {
        if tokens.is_empty() { return; }

        let kw = tokens[0].to_uppercase();
        let ch = kw.chars().next().unwrap_or(' ');

        match ch {
            'R' => {
                if tokens.len() >= 4 {
                    let a = self.resolve_node(tokens[1], prefix);
                    let b = self.resolve_node(tokens[2], prefix);
                    let val = self.parse_value(tokens[3]);
                    self.elements.push(Element::Resistor { a, b, r: val });
                }
            }
            'C' => {
                if tokens.len() >= 4 {
                    let a = self.resolve_node(tokens[1], prefix);
                    let b = self.resolve_node(tokens[2], prefix);
                    let val = self.parse_value(tokens[3]);
                    self.elements.push(Element::Capacitor { a, b, c: val });
                }
            }
            'L' => {
                // Inductor — for now we treat it like a zero-resistance wire in our linear solver.
                // Full inductor stamping (d/dt term) is deferred to Sprint 15.
                if tokens.len() >= 4 {
                    let a = self.resolve_node(tokens[1], prefix);
                    let b = self.resolve_node(tokens[2], prefix);
                    let _val = self.parse_value(tokens[3]);
                    // Approximation: short circuit (0Ω resistor) for DC operating point
                    self.elements.push(Element::Resistor { a, b, r: 1e-6 });
                }
            }
            'V' => {
                if tokens.len() >= 4 {
                    let pos = self.resolve_node(tokens[1], prefix);
                    let neg = self.resolve_node(tokens[2], prefix);

                    // Join the remainder so we can search across token boundaries
                    let remainder = tokens[3..].join(" ").to_uppercase();

                    let v = if remainder.starts_with("DC") {
                        // "DC 1.8" — skip keyword, parse next value
                        tokens.get(4).map(|s| self.parse_value(s)).unwrap_or(0.0)
                    } else if remainder.contains("PULSE(") || remainder.contains("PULSE (") {
                        // PULSE(V_lo V_hi td tr tf pw per)
                        // We extract V_hi (index 1 inside parens) as the steady-state
                        // stimulus for the DC operating point.
                        let after_paren = remainder
                            .splitn(2, '(')
                            .nth(1)
                            .unwrap_or("");
                        let vals: Vec<&str> = after_paren
                            .split(|c: char| c == ' ' || c == ',' || c == ')')
                            .filter(|s| !s.is_empty())
                            .collect();
                        // vals[0]=V_lo, vals[1]=V_hi
                        vals.get(1)
                            .and_then(|s| parse_scale_factor(s))
                            .unwrap_or(0.0)
                    } else if remainder.starts_with("SIN(") || remainder.starts_with("SIN (") {
                        // SIN(offset amplitude freq ...) — use amplitude as DC approximation
                        let after_paren = remainder
                            .splitn(2, '(')
                            .nth(1)
                            .unwrap_or("");
                        let vals: Vec<&str> = after_paren
                            .split(|c: char| c == ' ' || c == ',' || c == ')')
                            .filter(|s| !s.is_empty())
                            .collect();
                        // vals[0]=offset, vals[1]=amplitude
                        let offset = vals.first().and_then(|s| parse_scale_factor(s)).unwrap_or(0.0);
                        let amp = vals.get(1).and_then(|s| parse_scale_factor(s)).unwrap_or(0.0);
                        offset + amp
                    } else {
                        self.parse_value(tokens[3])
                    };

                    self.elements.push(Element::VoltageSource { pos, neg, v });
                }
            }
            'I' => {
                if tokens.len() >= 4 {
                    let pos = self.resolve_node(tokens[1], prefix);
                    let neg = self.resolve_node(tokens[2], prefix);
                    let i = self.parse_value(tokens[3]);
                    self.elements.push(Element::CurrentSource { pos, neg, i });
                }
            }
            'M' => {
                // MOSFET: Mname drain gate source bulk MODEL [W=.. L=..]
                if tokens.len() >= 6 {
                    let d = self.resolve_node(tokens[1], prefix);
                    let g = self.resolve_node(tokens[2], prefix);
                    let s = self.resolve_node(tokens[3], prefix);
                    let b = self.resolve_node(tokens[4], prefix);
                    let model_name = tokens[5].to_uppercase();

                    // Parse optional W= L= parameters
                    let mut w = 1e-6; // 1µm default width
                    let mut l = 100e-9; // 100nm default length
                    for tok in &tokens[6..] {
                        if let Some(eq) = tok.find('=') {
                            let key = tok[..eq].to_lowercase();
                            let val = self.parse_value(&tok[eq+1..]);
                            match key.as_str() {
                                "w" => w = val,
                                "l" => l = val,
                                _ => {}
                            }
                        }
                    }

                    // Look up parameters from .MODEL card or use defaults
                    let mut vth = 0.5;
                    let mut u0 = 0.04; // 400 cm^2/Vs ~ 0.04 m^2/Vs typical NMOS
                    let mut tox = 2e-9; // 2nm typical
                    
                    if let Some(model) = self.models.get(&model_name) {
                        if let Some(&v) = model.params.get("vto") { vth = v; }
                        if let Some(&u) = model.params.get("u0") { u0 = u * 1e-4; } // convert cm^2/Vs to m^2/Vs
                        if let Some(&t) = model.params.get("tox") { tox = t; }
                    }

                    let eps_ox = 3.9 * 8.854e-12;
                    let cox = eps_ox / tox;
                    let beta = u0 * cox * (w / l);

                    let model = MosfetLevel1 {
                        vth,
                        beta,
                        lambda: 0.1, // Fixed channel length modulation for now
                        node_g: g,
                        node_d: d,
                        node_s: s,
                    };
                    self.elements.push(Element::Mosfet { d, g, s, b, model });
                }
            }
            'D' => {
                // Diode: Dname anode cathode MODEL [AREA=..]
                if tokens.len() >= 4 {
                    let anode  = self.resolve_node(tokens[1], prefix);
                    let cathode = self.resolve_node(tokens[2], prefix);
                    let model_name = tokens[3].to_uppercase();
                    
                    let mut is = 1e-14;
                    let mut n = 1.0;
                    if let Some(model) = self.models.get(&model_name) {
                        if let Some(&v) = model.params.get("is") { is = v; }
                        if let Some(&v) = model.params.get("n") { n = v; }
                    }
                    
                    let model = DiodeModel {
                        is,
                        vt: 0.02585,
                        n,
                        node_p: anode,
                        node_n: cathode,
                    };
                    
                    self.elements.push(Element::Diode { p: anode, n: cathode, model });
                }
            }
            'X' => {
                // Subcircuit Instantiation: Xname port1 port2 ... subckt_name
                if tokens.len() > 2 {
                    let subckt_name = tokens.last().unwrap().to_uppercase();
                    
                    // Find definition
                    let def = match self.subcircuits.get(&subckt_name).cloned() {
                        Some(d) => d,
                        None => {
                            error!("Unknown subcircuit: {}", subckt_name);
                            return;
                        }
                    };

                    let instance_name = &tokens[0];
                    let instance_ports = &tokens[1..tokens.len() - 1];

                    if def.ports.len() != instance_ports.len() {
                        error!("Port mismatch in subcircuit {}: expected {}, got {}", subckt_name, def.ports.len(), instance_ports.len());
                        return;
                    }

                    // Create port mappings: formal port name → caller's actual node name.
                    // NOTE: We always store the raw actual name, never pre-prefixed.
                    // The actual node is in the *caller's* scope and is already registered globally.
                    let mut port_aliases: HashMap<String, String> = HashMap::new();
                    for (formal, actual) in def.ports.iter().zip(instance_ports.iter()) {
                        // Register the actual node into global map now, in case it hasn't been seen yet
                        self.resolve_node(actual, prefix);
                        let resolved_actual = if prefix.is_empty() {
                            actual.to_lowercase()
                        } else if self.node_map.contains_key(&actual.to_lowercase()) {
                            actual.to_lowercase() // Already global
                        } else {
                            format!("{}.{}", prefix, actual.to_lowercase())
                        };
                        port_aliases.insert(formal.to_lowercase(), resolved_actual);
                    }

                    // Recursive expansion: stream through the subcircuit's lines, substituting port variables
                    let new_prefix = if prefix.is_empty() { instance_name.to_string() } else { format!("{}.{}", prefix, instance_name) };
                    
                    for line in &def.lines {
                        let inner_tokens: Vec<&str> = line.split_whitespace().collect();
                        
                        let substituted_tokens: Vec<String> = inner_tokens.iter().map(|&t| {
                            let tl = t.to_lowercase();
                            if let Some(mapped) = port_aliases.get(&tl) {
                                mapped.clone()
                            } else {
                                t.to_string() // Must NOT recursively apply outer prefix here, handled by parse_element_line!
                            }
                        }).collect();
                        
                        let refs: Vec<&str> = substituted_tokens.iter().map(|s| s.as_str()).collect();
                        self.parse_element_line(&refs, &new_prefix);
                    }
                }
            }
            _ => {
                // Log warnings for directives that affect simulation but aren't yet handled
                let upper = kw.as_str();
                match upper {
                    ".INCLUDE" | ".LIB" => {
                        let file_ref = tokens.get(1).unwrap_or(&"<unknown>");
                        tracing::warn!(
                            "{} '{}' encountered but file inclusion is not yet supported. \
                             Models/subcircuits from this file will be missing.",
                            upper, file_ref
                        );
                    }
                    ".IC" | ".TRAN" | ".OPTIONS" | ".GLOBAL" | ".END" | ".ENDS" => {
                        // Known-safe directives we deliberately skip at this stage
                        debug!("Skipping directive: {}", upper);
                    }
                    ".GHOST" => {
                        // .GHOST <net_name> <local_index> <owner_partition>
                        if tokens.len() >= 4 {
                            let net_name = tokens[1];
                            let local_idx = tokens[2].parse::<usize>().unwrap_or(0);
                            let owner = tokens[3].parse::<u32>().unwrap_or(0);
                            
                            // Use a consistent hash for net_id across partitions
                            let net_id = self.hash_net_name(net_name);
                            self.ghost_map.push((net_id, local_idx, owner));
                            info!("Ghost node: {} (id: {}) maps to local node {}, owned by P{}", 
                                net_name, net_id, local_idx, owner);
                        }
                    }
                    ".NODEMAP" => {
                        // .NODEMAP <node_name> <index>
                        if tokens.len() >= 3 {
                            let name = tokens[1].to_lowercase();
                            let idx = tokens[2].parse::<usize>().unwrap_or(0);
                            debug!("Node map: {} -> {}", name, idx);
                            self.node_map.insert(name, idx);
                            if idx >= self.next_node_id {
                                self.next_node_id = idx + 1;
                            }
                        }
                    }
                    _ if upper.starts_with('.') => {
                        debug!("Unknown directive ignored: {}", upper);
                    }
                    _ => {} // Silently ignore unrecognised element first-chars
                }
            }
        }
    }

    /// Resolve a consistent hash for a net name across all partitions.
    fn hash_net_name(&self, name: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut s = DefaultHasher::new();
        name.to_lowercase().hash(&mut s);
        s.finish()
    }

    /// Resolve a generic node name to an ID. Prepend prefix if the node is internal to a subcircuit.
    /// E.g. node "n1" in "X1.X2" -> "X1.X2.n1"
    fn resolve_node(&mut self, name: &str, prefix: &str) -> usize {
        let name_lower = name.to_lowercase();
        // GND is strictly 0 and global across all subcircuits
        if name_lower == "0" || name_lower == "gnd" {
            return 0;
        }

        let full_name = if prefix.is_empty() {
            name_lower
        } else if name_lower.contains('.') || self.node_map.contains_key(&name_lower) {
            // Already fully-qualified: either has a '.' prefix separator, or was
            // registered globally before we entered this subcircuit scope (e.g. a
            // caller port like "1", "2", or "vdd").
            name_lower
        } else {
            format!("{}.{}", prefix, name_lower) // Internal isolated subcircuit node
        };

        self.get_or_create_node(&full_name)
    }

    /// Read an arbitrary SPICE .sp file and convert it into the solver's `PartitionNetlist`.
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<PartitionNetlist, std::io::Error> {
        let mut parser = SpiceParser::new();
        let path_ref = path.as_ref();

        info!("Parsing netlist: {:?}", path_ref);
        
        // Pass 1: Stash subcircuits
        parser.discover_subcircuits(path_ref)?;
        
        // Pass 2: Inflate instances and primitives
        parser.flatten_netlist(path_ref)?;

        // Pass 3: Global GMIN injection (100 MΩ pull-down on every mapped node)
        // This guarantees no floating nets exist anywhere, preventing Singular Matrix panics.
        let num_actual_nodes = parser.next_node_id; // Starts at 1
        for i in 1..num_actual_nodes {
            parser.elements.push(Element::Resistor { a: i, b: 0, r: 100e6 });
        }

        let name = path_ref.file_name().unwrap_or_default().to_string_lossy().into_owned();

        Ok(PartitionNetlist {
            name,
            num_nodes: parser.next_node_id - 1,
            elements: parser.elements,
            ghost_map: parser.ghost_map,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_scale_factors() {
        assert!((parse_scale_factor("1k").unwrap() - 1000.0).abs() < 1e-12);
        assert!((parse_scale_factor("5.5Meg").unwrap() - 5.5e6).abs() < 1e-12);
        assert!((parse_scale_factor("10f").unwrap() - 10e-15).abs() < 1e-20);
        assert!((parse_scale_factor("42").unwrap() - 42.0).abs() < 1e-12);
        assert!((parse_scale_factor("100p").unwrap() - 100e-12).abs() < 1e-20);
        assert!((parse_scale_factor("0").unwrap() - 0.0).abs() < 1e-12);
        assert!((parse_scale_factor("1.23u").unwrap() - 1.23e-6).abs() < 1e-12);
    }

    #[test]
    fn test_parse_simple_rc() {
        let test_netlist = r#"
        * Simple RC Circuit
        V1 1 0 1.8
        R1 1 2 1k
        C1 2 0 100f
        "#;

        let path = std::path::PathBuf::from("/tmp/nangila_test_rc.sp");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(test_netlist.as_bytes()).unwrap();

        let netlist = SpiceParser::parse_file(&path).unwrap();
        
        assert_eq!(netlist.num_nodes, 2);
        assert_eq!(netlist.elements.len(), 3);
        
        // Ensure R1 is 1000 ohms
        if let Element::Resistor { a: _, b: _, r } = netlist.elements[1] {
            assert_eq!(r, 1000.0);
        } else { panic!("Expected Resistor"); }
    }

    #[test]
    fn test_parse_subcircuits() {
        // Deep hierarchy test: File has a nested subcircuit call
        let test_netlist = r#"
        * Subcircuit Test

        .SUBCKT RC_STAGE in out
        R_s in internal 1k
        C_s internal 0 10f
        R_d internal out 500
        .ENDS

        V1 1 0 1.8
        * Instantiate RC_STAGE between node 1 and 2
        X1 1 2 RC_STAGE
        * Instantiate RC_STAGE between node 2 and 3
        X2 2 3 RC_STAGE

        R_load 3 0 100
        "#;

        let path = std::path::PathBuf::from("/tmp/nangila_test_subckt.sp");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(test_netlist.as_bytes()).unwrap();

        let netlist = SpiceParser::parse_file(&path).unwrap();
        
        // Primitives: 1 Vsource, 1 R_load
        // Instances: X1 expands to 2 Resistors, 1 Cap (3 elements)
        //            X2 expands to 2 Resistors, 1 Cap (3 elements)
        // Total elements expected: 1 + 1 + 3 + 3 = 8
        assert_eq!(netlist.elements.len(), 8);

        // Nodes: global 1, 2, 3
        //        internal X1.internal
        //        internal X2.internal
        // Total nodes: 5 (ignoring GND 0)
        assert_eq!(netlist.num_nodes, 5);
    }

    #[test]
    fn test_line_continuations() {
        let test_buf = b"R1 1 2\n+ 1k\nC1\n+ 2\n+ 0\n+ 100f";
        let reader = BufReader::new(&test_buf[..]);
        let lines: Vec<String> = SpiceParser::line_continuation_iter(reader).collect();
        
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "R1 1 2 1k");
        assert_eq!(lines[1], "C1 2 0 100f");
    }
}
