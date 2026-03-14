//! nangila-profile: Standalone profiler for mask calibration
//!
//! Commands:
//!   calibrate  Run calibration on recorded gradients to generate mask
//!   report     Generate topology report from an existing mask

use clap::{Parser, Subcommand};
use nangila_core::{Sculptor, Tensor, TopologyMask, TopologyReport};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nangila-profile")]
#[command(version, about = "Nangila topology profiler and mask generator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run calibration on gradient data to generate a topology mask
    Calibrate {
        /// Path to gradient data file (JSON lines format)
        #[arg(short, long)]
        gradients: PathBuf,

        /// Number of calibration steps to use (0 = all)
        #[arg(short, long, default_value = "500")]
        steps: usize,

        /// Correlation threshold for Passenger detection
        #[arg(short, long, default_value = "0.95")]
        threshold: f32,

        /// Output mask file path
        #[arg(short, long)]
        output: PathBuf,

        /// Also generate topology report
        #[arg(long)]
        report: Option<PathBuf>,
    },

    /// Generate topology report from an existing mask
    Report {
        /// Path to mask file
        #[arg(short, long)]
        mask: PathBuf,

        /// Output report file path (JSON)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Show summary of a mask file
    Info {
        /// Path to mask file
        #[arg(short, long)]
        mask: PathBuf,
    },
}

/// Gradient sample in JSON lines format
#[derive(Debug, Serialize, Deserialize)]
struct GradientSample {
    step: usize,
    layer_id: u32,
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Calibrate {
            gradients,
            steps,
            threshold,
            output,
            report,
        } => {
            run_calibrate(&gradients, steps, threshold, &output, report.as_ref())?;
        }
        Commands::Report { mask, output } => {
            run_report(&mask, &output)?;
        }
        Commands::Info { mask } => {
            run_info(&mask)?;
        }
    }

    Ok(())
}

fn run_calibrate(
    gradients_path: &PathBuf,
    max_steps: usize,
    threshold: f32,
    output_path: &PathBuf,
    report_path: Option<&PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Calibrating mask from: {}", gradients_path.display());
    println!("  Threshold: {:.2}", threshold);
    let max_steps_str = if max_steps == 0 {
        "all".to_string()
    } else {
        max_steps.to_string()
    };
    println!("  Max steps: {}", max_steps_str);

    let mut sculptor = Sculptor::new(threshold);

    // Read gradient samples
    let file = File::open(gradients_path)?;
    let reader = BufReader::new(file);
    let mut samples_read = 0;
    let mut current_step = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let sample: GradientSample = serde_json::from_str(&line)?;

        // Track step changes
        if sample.step != current_step {
            current_step = sample.step;
            if max_steps > 0 && current_step >= max_steps {
                break;
            }
        }

        let tensor = Tensor::new(sample.data, sample.shape);
        sculptor.record(sample.layer_id, &tensor);
        samples_read += 1;
    }

    println!("  Read {} gradient samples", samples_read);
    println!("  Tracking {} layers", sculptor.num_layers());

    // Generate mask
    let mask = sculptor.generate_mask()?;

    println!("\nGenerated mask:");
    println!("  Drivers:     {}", mask.num_drivers());
    println!("  Passengers:  {}", mask.num_passengers());
    println!("  Compression: {:.2}x", mask.compression_ratio());

    // Save mask
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    mask.save(&mut writer)?;
    println!("\nMask saved to: {}", output_path.display());

    // Optionally generate report
    if let Some(report_path) = report_path {
        let report = TopologyReport::from_sculptor_and_mask(&sculptor, &mask);
        let json = report.to_json();
        std::fs::write(report_path, &json)?;
        println!("Report saved to: {}", report_path.display());
    }

    Ok(())
}

fn run_report(
    mask_path: &PathBuf,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading mask from: {}", mask_path.display());

    let file = File::open(mask_path)?;
    let mut reader = BufReader::new(file);
    let mask = TopologyMask::load(&mut reader)?;

    let report = TopologyReport::from_mask(&mask);
    let json = report.to_json();

    std::fs::write(output_path, &json)?;
    println!("Report saved to: {}", output_path.display());

    Ok(())
}

fn run_info(mask_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(mask_path)?;
    let mut reader = BufReader::new(file);
    let mask = TopologyMask::load(&mut reader)?;

    println!("Mask: {}", mask_path.display());
    println!("  Total layers:  {}", mask.num_layers());
    println!("  Drivers:       {}", mask.num_drivers());
    println!("  Passengers:    {}", mask.num_passengers());
    println!("  Compression:   {:.2}x", mask.compression_ratio());

    Ok(())
}
