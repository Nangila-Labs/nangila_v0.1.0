//! Build script for CUDA kernel compilation
//!
//! This script compiles .cu files using nvcc and links them with the Rust crate.
//! Set CUDA_PATH environment variable if CUDA is not in the default location.

use std::env;
use std::path::PathBuf;

fn main() {
    // Check if CUDA is available
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = PathBuf::from(&cuda_path).join("include");
    let cuda_lib = PathBuf::from(&cuda_path).join("lib64");

    // Check if CUDA exists
    if !cuda_include.exists() {
        println!(
            "cargo:warning=CUDA not found at {}. Using CPU fallback.",
            cuda_path
        );
        println!("cargo:rustc-cfg=feature=\"cpu_fallback\"");
        return;
    }

    println!("cargo:rustc-cfg=feature=\"cuda\"");

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");

    // Compile CUDA kernels
    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("kernels");

    let cu_files = ["predict.cu", "reconstruct.cu", "gamma.cu"];
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    for cu_file in &cu_files {
        let cu_path = kernel_dir.join(cu_file);
        if !cu_path.exists() {
            println!("cargo:warning=Kernel file {} not found", cu_file);
            continue;
        }

        let obj_name = cu_file.replace(".cu", ".o");
        let obj_path = out_dir.join(&obj_name);

        // Compile with nvcc
        let status = std::process::Command::new("nvcc")
            .args(&[
                "-c",
                "-O3",
                "--use_fast_math",
                "-gencode",
                "arch=compute_70,code=sm_70", // V100
                "-gencode",
                "arch=compute_80,code=sm_80", // A100
                "-gencode",
                "arch=compute_89,code=sm_89", // H100
                "-I",
                cuda_include.to_str().unwrap(),
                "-o",
                obj_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rerun-if-changed={}", cu_path.display());
            }
            Ok(s) => {
                println!(
                    "cargo:warning=nvcc failed for {} with exit code {:?}",
                    cu_file,
                    s.code()
                );
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to run nvcc: {}. Using CPU fallback.",
                    e
                );
                println!("cargo:rustc-cfg=feature=\"cpu_fallback\"");
                return;
            }
        }
    }

    // Create static library from object files
    let lib_path = out_dir.join("libnangila_kernels.a");
    let obj_files: Vec<_> = cu_files
        .iter()
        .map(|f| out_dir.join(f.replace(".cu", ".o")))
        .filter(|p| p.exists())
        .collect();

    if !obj_files.is_empty() {
        let status = std::process::Command::new("ar")
            .args(&["rcs", lib_path.to_str().unwrap()])
            .args(obj_files.iter().map(|p| p.to_str().unwrap()))
            .status();

        if let Ok(s) = status {
            if s.success() {
                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=nangila_kernels");
            }
        }
    }
}
