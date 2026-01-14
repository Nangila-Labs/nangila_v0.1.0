//! Nangila Dump: Checkpoint Verification and Crash Forensics
//!
//! CLI tool for:
//! - Verifying checkpoint integrity
//! - Comparing checkpoints from different nodes
//! - Diagnosing desync issues
//! - Inspecting predictor state

use clap::{Parser, Subcommand};
use nangila_core::NangilaCheckpoint;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nangila-dump")]
#[command(about = "Checkpoint verification and crash forensics for Nangila")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify checkpoint integrity
    Verify {
        /// Path to checkpoint file
        #[arg(short, long)]
        checkpoint: PathBuf,
    },

    /// Compare two checkpoints (detect divergence)
    Diff {
        /// First checkpoint
        #[arg(short = 'a', long)]
        checkpoint_a: PathBuf,
        /// Second checkpoint
        #[arg(short = 'b', long)]
        checkpoint_b: PathBuf,
    },

    /// Inspect checkpoint contents as JSON
    Inspect {
        /// Path to checkpoint file
        #[arg(short, long)]
        checkpoint: PathBuf,
        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show summary of checkpoint
    Info {
        /// Path to checkpoint file
        #[arg(short, long)]
        checkpoint: PathBuf,
    },
}

/// JSON output for checkpoint inspection
#[derive(Serialize)]
struct CheckpointInfo {
    version: u32,
    step: u64,
    state_hash: String,
    integrity_valid: bool,
    num_layers: usize,
    quantizer_gamma: f32,
    warmup_steps: usize,
    predictor_step: usize,
    predictor_momentum: f32,
}

/// JSON output for layer state
#[derive(Serialize)]
struct LayerInfo {
    layer_id: u32,
    has_prev: bool,
    has_current: bool,
    prev_size: usize,
    current_size: usize,
    prev_hash: Option<String>,
    current_hash: Option<String>,
}

/// JSON output for diff results
#[derive(Serialize)]
struct DiffResult {
    match_status: String,
    step_a: u64,
    step_b: u64,
    hash_a: String,
    hash_b: String,
    divergent_layers: Vec<u32>,
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Verify { checkpoint } => cmd_verify(&checkpoint),
        Commands::Diff {
            checkpoint_a,
            checkpoint_b,
        } => cmd_diff(&checkpoint_a, &checkpoint_b),
        Commands::Inspect { checkpoint, output } => cmd_inspect(&checkpoint, output.as_deref()),
        Commands::Info { checkpoint } => cmd_info(&checkpoint),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_verify(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = NangilaCheckpoint::load(path)?;

    let computed_hash = checkpoint.compute_hash();
    let stored_hash = checkpoint.state_hash;
    let valid = computed_hash == stored_hash;

    if valid {
        println!("✓ VALID: Checkpoint integrity verified");
        println!("  Step: {}", checkpoint.step);
        println!("  Hash: {:016x}", stored_hash);
        println!("  Layers: {}", checkpoint.predictor.histories.len());
    } else {
        println!("✗ INVALID: Checkpoint integrity check failed");
        println!("  Stored hash:   {:016x}", stored_hash);
        println!("  Computed hash: {:016x}", computed_hash);
        std::process::exit(2);
    }

    Ok(())
}

fn cmd_diff(path_a: &PathBuf, path_b: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let chk_a = NangilaCheckpoint::load(path_a)?;
    let chk_b = NangilaCheckpoint::load(path_b)?;

    let hash_a = chk_a.compute_hash();
    let hash_b = chk_b.compute_hash();

    let mut divergent_layers = Vec::new();

    // Compare layer by layer
    for (&layer_id, hist_a) in &chk_a.predictor.histories {
        if let Some(hist_b) = chk_b.predictor.histories.get(&layer_id) {
            // Compare prev buffers
            if hist_a.prev != hist_b.prev {
                divergent_layers.push(layer_id);
                continue;
            }
            // Compare current buffers
            if hist_a.current != hist_b.current {
                divergent_layers.push(layer_id);
            }
        } else {
            divergent_layers.push(layer_id);
        }
    }

    // Check for layers only in B
    for &layer_id in chk_b.predictor.histories.keys() {
        if !chk_a.predictor.histories.contains_key(&layer_id) && !divergent_layers.contains(&layer_id) {
            divergent_layers.push(layer_id);
        }
    }

    divergent_layers.sort();

    let result = DiffResult {
        match_status: if divergent_layers.is_empty() {
            "MATCH".to_string()
        } else {
            "DIVERGED".to_string()
        },
        step_a: chk_a.step,
        step_b: chk_b.step,
        hash_a: format!("{:016x}", hash_a),
        hash_b: format!("{:016x}", hash_b),
        divergent_layers: divergent_layers.clone(),
    };

    println!("{}", serde_json::to_string_pretty(&result)?);

    if !divergent_layers.is_empty() {
        std::process::exit(3);
    }

    Ok(())
}

fn cmd_inspect(path: &PathBuf, output: Option<&std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = NangilaCheckpoint::load(path)?;

    let info = CheckpointInfo {
        version: checkpoint.version,
        step: checkpoint.step,
        state_hash: format!("{:016x}", checkpoint.state_hash),
        integrity_valid: checkpoint.verify_integrity(),
        num_layers: checkpoint.predictor.histories.len(),
        quantizer_gamma: checkpoint.quantizer_gamma,
        warmup_steps: checkpoint.config.warmup_steps,
        predictor_step: checkpoint.predictor.current_step,
        predictor_momentum: checkpoint.predictor.momentum,
    };

    let mut layers: Vec<LayerInfo> = checkpoint
        .predictor
        .histories
        .iter()
        .map(|(&layer_id, hist)| {
            let prev_hash = hist.prev.as_ref().map(|v| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                v.hash(&mut h);
                format!("{:016x}", h.finish())
            });
            let current_hash = hist.current.as_ref().map(|v| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                v.hash(&mut h);
                format!("{:016x}", h.finish())
            });

            LayerInfo {
                layer_id,
                has_prev: hist.prev.is_some(),
                has_current: hist.current.is_some(),
                prev_size: hist.prev.as_ref().map(|v| v.len()).unwrap_or(0),
                current_size: hist.current.as_ref().map(|v| v.len()).unwrap_or(0),
                prev_hash,
                current_hash,
            }
        })
        .collect();
    layers.sort_by_key(|l| l.layer_id);

    #[derive(Serialize)]
    struct FullInspection {
        checkpoint: CheckpointInfo,
        layers: Vec<LayerInfo>,
    }

    let full = FullInspection {
        checkpoint: info,
        layers,
    };

    let json = serde_json::to_string_pretty(&full)?;

    if let Some(out_path) = output {
        std::fs::write(out_path, &json)?;
        println!("Written to: {}", out_path.display());
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn cmd_info(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = NangilaCheckpoint::load(path)?;

    println!("Nangila Checkpoint: {}", path.display());
    println!("─────────────────────────────────────");
    println!("  Version:          {}", checkpoint.version);
    println!("  Step:             {}", checkpoint.step);
    println!("  State Hash:       {:016x}", checkpoint.state_hash);
    println!("  Integrity:        {}", if checkpoint.verify_integrity() { "✓ Valid" } else { "✗ Invalid" });
    println!("  Layers:           {}", checkpoint.predictor.histories.len());
    println!("  Quantizer Gamma:  {:.6}", checkpoint.quantizer_gamma);
    println!("  Predictor Step:   {}", checkpoint.predictor.current_step);
    println!("  Predictor μ:      {}", checkpoint.predictor.momentum);
    println!("  Warmup Steps:     {}", checkpoint.config.warmup_steps);

    Ok(())
}
