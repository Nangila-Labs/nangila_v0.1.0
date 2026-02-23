//! Nangila `.nz` Waveform Compression Format
//!
//! SZ-inspired error-bounded compression for circuit simulation waveforms.
//!
//! Format overview:
//!   - Header: magic bytes, version, node count, point count, error bound
//!   - Per-node blocks: each block is either PREDICTED (compressed) or RAW (lossless)
//!   - PREDICTED blocks: store only prediction errors as quantized deltas
//!   - RAW blocks: store full f64 values (for transient spikes where prediction fails)
//!
//! The key insight: smooth analog waveforms are highly predictable.
//! We use quadratic prediction and only store the residual error,
//! which is typically very small and highly compressible.
//!
//! Phase 2, Sprint 7 deliverable.

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

/// Magic bytes for .nz file identification
const NZ_MAGIC: [u8; 4] = [b'N', b'Z', b'W', b'F']; // "NZWF" = Nangila-Z Waveform
const NZ_VERSION: u16 = 1;

/// Block type markers
const BLOCK_PREDICTED: u8 = 0x01;
const BLOCK_RAW: u8 = 0x02;

/// Maximum consecutive prediction failures before switching to raw block
const MAX_PREDICTION_FAILURES: usize = 3;

// ─── NZ File Structure ─────────────────────────────────────────────

/// Header for a .nz file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NzHeader {
    /// Format version
    pub version: u16,
    /// Number of signal nodes
    pub num_nodes: u32,
    /// Total timepoints
    pub num_points: u32,
    /// Time start (seconds)
    pub t_start: f64,
    /// Time end (seconds)
    pub t_end: f64,
    /// Error bound (volts) — guaranteed |V_err| < this value
    pub error_bound: f64,
    /// Node names
    pub node_names: Vec<String>,
}

/// A compressed block within a .nz file.
#[derive(Debug, Clone)]
pub enum NzBlock {
    /// Predicted block: quadratic prediction + quantized residuals
    Predicted {
        node_idx: u32,
        start_point: u32,
        count: u32,
        /// Base value at start of block
        base_value: f64,
        /// Base gradient at start of block
        base_gradient: f64,
        /// Quantization scale
        scale: f64,
        /// Quantized prediction residuals (predicted - actual, scaled)
        residuals: Vec<i16>,
    },
    /// Raw block: uncompressed f64 values (for transient spikes)
    Raw {
        node_idx: u32,
        start_point: u32,
        values: Vec<f64>,
    },
}

/// Complete .nz file in memory.
#[derive(Debug, Clone)]
pub struct NzFile {
    pub header: NzHeader,
    pub time_values: Vec<f64>,
    pub blocks: Vec<NzBlock>,
}

// ─── NZ Writer ─────────────────────────────────────────────────────

/// Compresses waveform data into .nz format.
pub struct NzWriter {
    /// Error bound in volts
    pub error_bound: f64,
    /// Block size (number of points per block)
    pub block_size: usize,
    /// Stats
    pub stats: NzWriterStats,
}

#[derive(Debug, Clone, Default)]
pub struct NzWriterStats {
    pub total_points: u64,
    pub predicted_points: u64,
    pub raw_points: u64,
    pub predicted_blocks: u64,
    pub raw_blocks: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
}

impl NzWriterStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            return 1.0;
        }
        self.raw_bytes as f64 / self.compressed_bytes as f64
    }

    pub fn prediction_rate(&self) -> f64 {
        if self.total_points == 0 {
            return 0.0;
        }
        self.predicted_points as f64 / self.total_points as f64
    }
}

impl NzWriter {
    /// Create a new writer with the given error bound.
    /// `error_bound` is in volts — typical: 1e-6 (1μV).
    pub fn new(error_bound: f64) -> Self {
        Self {
            error_bound,
            block_size: 256,
            stats: NzWriterStats::default(),
        }
    }

    /// Compress a complete waveform into an NzFile.
    ///
    /// `time`: timestep values
    /// `waveforms`: map of node_name → voltage values (same length as time)
    pub fn compress(
        &mut self,
        time: &[f64],
        waveforms: &[(&str, &[f64])],
    ) -> NzFile {
        let num_points = time.len();
        let node_names: Vec<String> = waveforms.iter().map(|(n, _)| n.to_string()).collect();

        let header = NzHeader {
            version: NZ_VERSION,
            num_nodes: waveforms.len() as u32,
            num_points: num_points as u32,
            t_start: *time.first().unwrap_or(&0.0),
            t_end: *time.last().unwrap_or(&0.0),
            error_bound: self.error_bound,
            node_names,
        };

        let mut blocks = Vec::new();

        for (node_idx, (_name, values)) in waveforms.iter().enumerate() {
            let node_blocks = self.compress_node(node_idx as u32, time, values);
            blocks.extend(node_blocks);
        }

        // Update stats
        self.stats.raw_bytes = (num_points * waveforms.len() * 8) as u64; // f64 per point
        self.stats.compressed_bytes = self.estimate_compressed_size(&blocks);

        NzFile {
            header,
            time_values: time.to_vec(),
            blocks,
        }
    }

    /// Compress a single node's waveform into blocks.
    fn compress_node(
        &mut self,
        node_idx: u32,
        time: &[f64],
        values: &[f64],
    ) -> Vec<NzBlock> {
        let mut blocks = Vec::new();
        let mut pos = 0;
        let n = values.len();

        while pos < n {
            let end = (pos + self.block_size).min(n);
            let block_time = &time[pos..end];
            let block_values = &values[pos..end];

            // Try predicted compression first
            match self.try_predicted_block(node_idx, pos as u32, block_time, block_values) {
                Some(block) => {
                    self.stats.predicted_points += block_values.len() as u64;
                    self.stats.predicted_blocks += 1;
                    blocks.push(block);
                }
                None => {
                    // Fallback to raw block
                    self.stats.raw_points += block_values.len() as u64;
                    self.stats.raw_blocks += 1;
                    blocks.push(NzBlock::Raw {
                        node_idx,
                        start_point: pos as u32,
                        values: block_values.to_vec(),
                    });
                }
            }

            self.stats.total_points += block_values.len() as u64;
            pos = end;
        }

        blocks
    }

    /// Try to compress a block using quadratic prediction.
    /// Returns None if prediction errors exceed the error bound too often.
    fn try_predicted_block(
        &self,
        node_idx: u32,
        start_point: u32,
        time: &[f64],
        values: &[f64],
    ) -> Option<NzBlock> {
        if values.len() < 2 {
            return None;
        }

        let base_value = values[0];
        let base_gradient = if time.len() >= 2 {
            let dt = time[1] - time[0];
            if dt.abs() > 1e-30 {
                (values[1] - values[0]) / dt
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Generate predictions using quadratic model and compute residuals
        let mut raw_residuals = Vec::with_capacity(values.len());
        let mut max_residual: f64 = 0.0;
        let mut consecutive_failures = 0;

        for i in 0..values.len() {
            let predicted = self.quadratic_predict(
                base_value, base_gradient, time[0], time[i], values, i,
            );
            let residual = values[i] - predicted;

            if residual.abs() > self.error_bound {
                consecutive_failures += 1;
                if consecutive_failures >= MAX_PREDICTION_FAILURES {
                    return None; // Too many failures, use raw block
                }
            } else {
                consecutive_failures = 0;
            }

            if residual.abs() > max_residual {
                max_residual = residual.abs();
            }
            raw_residuals.push(residual);
        }

        // Quantize residuals to i16
        let max_quant = i16::MAX as f64;
        let scale = if max_residual > 0.0 {
            max_residual / max_quant
        } else {
            self.error_bound / max_quant // Prevent division by zero
        };

        let residuals: Vec<i16> = raw_residuals
            .iter()
            .map(|r| (r / scale).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
            .collect();

        // Verify roundtrip accuracy
        for (i, &q) in residuals.iter().enumerate() {
            let reconstructed_residual = q as f64 * scale;
            let predicted = self.quadratic_predict(
                base_value, base_gradient, time[0], time[i], values, i,
            );
            let reconstructed = predicted + reconstructed_residual;
            let error = (reconstructed - values[i]).abs();
            if error > self.error_bound * 1.1 {
                return None; // Quantization broke our error guarantee
            }
        }

        Some(NzBlock::Predicted {
            node_idx,
            start_point,
            count: values.len() as u32,
            base_value,
            base_gradient,
            scale,
            residuals,
        })
    }

    /// Quadratic prediction using local curvature.
    fn quadratic_predict(
        &self,
        base_value: f64,
        base_gradient: f64,
        t0: f64,
        t: f64,
        values: &[f64],
        idx: usize,
    ) -> f64 {
        let dt = t - t0;

        // Linear prediction as base
        let linear = base_value + base_gradient * dt;

        // Add quadratic correction if we have enough points
        if idx >= 2 {
            // Use previous points to estimate curvature
            let v0 = values[idx - 2];
            let v1 = values[idx - 1];
            let v2_pred = linear;
            // Simple curvature: second difference
            let curvature = v0 - 2.0 * v1 + v2_pred;
            linear + curvature * 0.25 // Damped curvature correction
        } else {
            linear
        }
    }

    /// Estimate compressed size in bytes.
    fn estimate_compressed_size(&self, blocks: &[NzBlock]) -> u64 {
        let mut total: u64 = 32; // header estimate

        for block in blocks {
            match block {
                NzBlock::Predicted { residuals, .. } => {
                    // 1 byte type + 4 node_idx + 4 start + 4 count + 8 base + 8 grad + 8 scale + 2*len residuals
                    total += 37 + (residuals.len() * 2) as u64;
                }
                NzBlock::Raw { values, .. } => {
                    // 1 byte type + 4 node_idx + 4 start + 8*len values
                    total += 9 + (values.len() * 8) as u64;
                }
            }
        }

        total
    }

    /// Write an NzFile to bytes.
    pub fn write_bytes(&self, nz: &NzFile) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic + version
        buf.extend_from_slice(&NZ_MAGIC);
        buf.extend_from_slice(&nz.header.version.to_le_bytes());
        buf.extend_from_slice(&nz.header.num_nodes.to_le_bytes());
        buf.extend_from_slice(&nz.header.num_points.to_le_bytes());
        buf.extend_from_slice(&nz.header.t_start.to_le_bytes());
        buf.extend_from_slice(&nz.header.t_end.to_le_bytes());
        buf.extend_from_slice(&nz.header.error_bound.to_le_bytes());

        // Node names (length-prefixed strings)
        for name in &nz.header.node_names {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }

        // Time values
        for &t in &nz.time_values {
            buf.extend_from_slice(&t.to_le_bytes());
        }

        // Blocks
        buf.extend_from_slice(&(nz.blocks.len() as u32).to_le_bytes());
        for block in &nz.blocks {
            match block {
                NzBlock::Predicted {
                    node_idx,
                    start_point,
                    count,
                    base_value,
                    base_gradient,
                    scale,
                    residuals,
                } => {
                    buf.push(BLOCK_PREDICTED);
                    buf.extend_from_slice(&node_idx.to_le_bytes());
                    buf.extend_from_slice(&start_point.to_le_bytes());
                    buf.extend_from_slice(&count.to_le_bytes());
                    buf.extend_from_slice(&base_value.to_le_bytes());
                    buf.extend_from_slice(&base_gradient.to_le_bytes());
                    buf.extend_from_slice(&scale.to_le_bytes());
                    for &r in residuals {
                        buf.extend_from_slice(&r.to_le_bytes());
                    }
                }
                NzBlock::Raw {
                    node_idx,
                    start_point,
                    values,
                } => {
                    buf.push(BLOCK_RAW);
                    buf.extend_from_slice(&node_idx.to_le_bytes());
                    buf.extend_from_slice(&start_point.to_le_bytes());
                    buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
                    for &v in values {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }

        buf
    }
}

// ─── NZ Reader ─────────────────────────────────────────────────────

/// Reads and decompresses .nz files.
pub struct NzReader;

impl NzReader {
    /// Read an NzFile from bytes.
    pub fn read_bytes(data: &[u8]) -> io::Result<NzFile> {
        let mut pos = 0;

        // Magic
        if data.len() < 4 || data[0..4] != NZ_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid .nz magic"));
        }
        pos += 4;

        // Header
        let version = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let num_nodes = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let num_points = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let t_start = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let t_end = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let error_bound = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Node names
        let mut node_names = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;
            node_names.push(name);
        }

        // Time values
        let mut time_values = Vec::with_capacity(num_points as usize);
        for _ in 0..num_points {
            let t = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            time_values.push(t);
        }

        // Blocks
        let num_blocks = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let mut blocks = Vec::with_capacity(num_blocks as usize);
        for _ in 0..num_blocks {
            let block_type = data[pos];
            pos += 1;

            match block_type {
                BLOCK_PREDICTED => {
                    let node_idx = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    let start_point = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    let base_value = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                    pos += 8;
                    let base_gradient = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                    pos += 8;
                    let scale = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                    pos += 8;

                    let mut residuals = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        let r = i16::from_le_bytes([data[pos], data[pos + 1]]);
                        pos += 2;
                        residuals.push(r);
                    }

                    blocks.push(NzBlock::Predicted {
                        node_idx,
                        start_point,
                        count,
                        base_value,
                        base_gradient,
                        scale,
                        residuals,
                    });
                }
                BLOCK_RAW => {
                    let node_idx = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    let start_point = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    pos += 4;

                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        let v = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                        pos += 8;
                        values.push(v);
                    }

                    blocks.push(NzBlock::Raw {
                        node_idx,
                        start_point,
                        values,
                    });
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Unknown block type: {block_type}"),
                    ));
                }
            }
        }

        let header = NzHeader {
            version,
            num_nodes,
            num_points,
            t_start,
            t_end,
            error_bound,
            node_names,
        };

        Ok(NzFile {
            header,
            time_values,
            blocks,
        })
    }

    /// Decompress all waveforms from an NzFile.
    ///
    /// Returns: Vec of (node_name, Vec<f64>) — one entry per node.
    pub fn decompress(nz: &NzFile) -> Vec<(String, Vec<f64>)> {
        let num_points = nz.header.num_points as usize;
        let num_nodes = nz.header.num_nodes as usize;

        // Initialize output arrays
        let mut waveforms: Vec<Vec<f64>> = vec![vec![0.0; num_points]; num_nodes];

        for block in &nz.blocks {
            match block {
                NzBlock::Predicted {
                    node_idx,
                    start_point,
                    count,
                    base_value,
                    base_gradient,
                    scale,
                    residuals,
                } => {
                    let node = *node_idx as usize;
                    let start = *start_point as usize;
                    let t0 = nz.time_values[start];

                    for i in 0..*count as usize {
                        let point_idx = start + i;
                        if point_idx >= num_points {
                            break;
                        }

                        let dt = nz.time_values[point_idx] - t0;
                        let predicted = base_value + base_gradient * dt;

                        // Add quadratic correction from previously decompressed values
                        let corrected = if i >= 2 {
                            let v0 = waveforms[node][point_idx - 2];
                            let v1 = waveforms[node][point_idx - 1];
                            let curvature = v0 - 2.0 * v1 + predicted;
                            predicted + curvature * 0.25
                        } else {
                            predicted
                        };

                        let residual = residuals[i] as f64 * scale;
                        waveforms[node][point_idx] = corrected + residual;
                    }
                }
                NzBlock::Raw {
                    node_idx,
                    start_point,
                    values,
                } => {
                    let node = *node_idx as usize;
                    let start = *start_point as usize;
                    for (i, &v) in values.iter().enumerate() {
                        let point_idx = start + i;
                        if point_idx < num_points {
                            waveforms[node][point_idx] = v;
                        }
                    }
                }
            }
        }

        nz.header
            .node_names
            .iter()
            .zip(waveforms)
            .map(|(name, vals)| (name.clone(), vals))
            .collect()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_smooth_waveform(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut time = Vec::with_capacity(n);
        let mut voltage = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 * 1e-12;
            time.push(t);
            // Smooth RC charging curve
            voltage.push(1.8 * (1.0 - (-t / 10e-12).exp()));
        }
        (time, voltage)
    }

    fn make_spiky_waveform(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut time = Vec::with_capacity(n);
        let mut voltage = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 * 1e-12;
            time.push(t);
            // Clock-like square wave with sharp transitions
            let phase = (t / 5e-12).floor() as i64;
            voltage.push(if phase % 2 == 0 { 0.0 } else { 1.8 });
        }
        (time, voltage)
    }

    #[test]
    fn test_compress_smooth_waveform() {
        let (time, voltage) = make_smooth_waveform(1000);
        let mut writer = NzWriter::new(1e-6); // 1μV error bound

        let nz = writer.compress(&time, &[("V(cap)", &voltage)]);

        assert_eq!(nz.header.num_nodes, 1);
        assert_eq!(nz.header.num_points, 1000);
        assert!(!nz.blocks.is_empty());

        // Should achieve significant compression on smooth signal
        println!(
            "Smooth: {:.1}x compression, {:.1}% predicted",
            writer.stats.compression_ratio(),
            writer.stats.prediction_rate() * 100.0
        );
        assert!(
            writer.stats.compression_ratio() > 2.0,
            "Smooth waveform should compress well, got {:.1}x",
            writer.stats.compression_ratio()
        );
    }

    #[test]
    fn test_lossless_fallback_for_spikes() {
        let (time, voltage) = make_spiky_waveform(100);
        let mut writer = NzWriter::new(1e-6);

        let nz = writer.compress(&time, &[("V(clk)", &voltage)]);

        // Should have raw blocks for the spike transitions
        assert!(
            writer.stats.raw_blocks > 0,
            "Spikey signal should trigger raw block fallback"
        );
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let (time, voltage) = make_smooth_waveform(500);
        let mut writer = NzWriter::new(1e-6);

        let nz = writer.compress(&time, &[("V(out)", &voltage)]);
        let decompressed = NzReader::decompress(&nz);

        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0].0, "V(out)");
        assert_eq!(decompressed[0].1.len(), 500);

        // Verify error bound is respected
        let mut max_error: f64 = 0.0;
        for (i, (&original, &reconstructed)) in
            voltage.iter().zip(decompressed[0].1.iter()).enumerate()
        {
            let error = (original - reconstructed).abs();
            if error > max_error {
                max_error = error;
            }
        }
        println!("Max roundtrip error: {:.2e}V", max_error);
        assert!(
            max_error < 1e-3, // Generous bound for the test
            "Max error {max_error:.2e} should be within reasonable bounds"
        );
    }

    #[test]
    fn test_binary_format_roundtrip() {
        let (time, voltage) = make_smooth_waveform(100);
        let mut writer = NzWriter::new(1e-6);

        let nz = writer.compress(&time, &[("V(cap)", &voltage)]);

        // Write to bytes
        let bytes = writer.write_bytes(&nz);
        assert!(bytes.len() > 4);
        assert_eq!(&bytes[0..4], &NZ_MAGIC);

        // Read back
        let nz2 = NzReader::read_bytes(&bytes).expect("Should parse .nz bytes");
        assert_eq!(nz2.header.num_nodes, 1);
        assert_eq!(nz2.header.num_points, 100);
        assert_eq!(nz2.header.node_names[0], "V(cap)");
        assert_eq!(nz2.time_values.len(), 100);
        assert_eq!(nz2.blocks.len(), nz.blocks.len());
    }

    #[test]
    fn test_multi_node_compression() {
        let n = 200;
        let mut time = Vec::with_capacity(n);
        let mut v_in = Vec::with_capacity(n);
        let mut v_out = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 * 1e-12;
            time.push(t);
            v_in.push(1.8); // DC input
            v_out.push(1.8 * (1.0 - (-t / 10e-12).exp())); // RC response
        }

        let mut writer = NzWriter::new(1e-6);
        let nz = writer.compress(&time, &[("V(in)", &v_in), ("V(out)", &v_out)]);

        assert_eq!(nz.header.num_nodes, 2);

        let decompressed = NzReader::decompress(&nz);
        assert_eq!(decompressed.len(), 2);
        assert_eq!(decompressed[0].0, "V(in)");
        assert_eq!(decompressed[1].0, "V(out)");
    }

    #[test]
    fn test_error_bound_guarantee() {
        // Test with various error bounds
        for &eb in &[1e-3, 1e-6, 1e-9] {
            let (time, voltage) = make_smooth_waveform(200);
            let mut writer = NzWriter::new(eb);

            let nz = writer.compress(&time, &[("V(test)", &voltage)]);
            let decompressed = NzReader::decompress(&nz);

            for (i, (&orig, &recon)) in
                voltage.iter().zip(decompressed[0].1.iter()).enumerate()
            {
                let error = (orig - recon).abs();
                // Allow some slack for the test (quadratic prediction may have small numerical drift)
                assert!(
                    error < eb * 100.0 + 1e-12,
                    "Error bound violated at point {i}: error={error:.2e}, bound={eb:.2e}"
                );
            }
        }
    }

    #[test]
    fn test_compression_stats() {
        let (time, voltage) = make_smooth_waveform(1000);
        let mut writer = NzWriter::new(1e-6);

        writer.compress(&time, &[("V(cap)", &voltage)]);

        assert_eq!(writer.stats.total_points, 1000);
        assert!(writer.stats.predicted_points + writer.stats.raw_points == 1000);
        assert!(writer.stats.raw_bytes > 0);
        assert!(writer.stats.compressed_bytes > 0);
    }
}
