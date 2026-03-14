//! Build script for CUDA kernel compilation
//!
//! This script compiles .cu files using nvcc and links them with the Rust crate.
//! Set CUDA_PATH environment variable if CUDA is not in the default location.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=NANGILA_CUDA_ARCHES");

    let cuda_feature_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();

    // Check if CUDA is available
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_root = PathBuf::from(&cuda_path);
    let standard_include = cuda_root.join("include");
    let standard_lib = cuda_root.join("lib64");
    let targets_include = cuda_root.join("targets/x86_64-linux/include");
    let targets_lib = cuda_root.join("targets/x86_64-linux/lib");

    let (cuda_include, cuda_lib) = if standard_include.exists() && standard_lib.exists() {
        (standard_include, standard_lib)
    } else if targets_include.exists() && targets_lib.exists() {
        (targets_include, targets_lib)
    } else {
        if cuda_feature_enabled {
            panic!(
                "CUDA feature was requested, but no usable CUDA include/lib layout was found under {}. \
Expected either include/ + lib64/ or targets/x86_64-linux/include + targets/x86_64-linux/lib.",
                cuda_path
            );
        }
        return;
    };

    // Check if CUDA exists
    if !cuda_include.exists() {
        if cuda_feature_enabled {
            panic!(
                "CUDA feature was requested, but CUDA headers were not found at {}. \
Set CUDA_HOME/CUDA_PATH to a valid toolkit root and ensure nvcc is available.",
                cuda_path
            );
        }
        return;
    }

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");

    // Compile CUDA kernels
    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("kernels");

    let cu_files = ["predict.cu", "reconstruct.cu", "gamma.cu", "crc32.cu"];
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_arches = env::var("NANGILA_CUDA_ARCHES").unwrap_or_else(|_| "80,89,90".to_string());
    let gencode_args: Vec<String> = cuda_arches
        .split(',')
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
        .flat_map(|arch| {
            [
                "-gencode".to_string(),
                format!("arch=compute_{arch},code=sm_{arch}"),
            ]
        })
        .collect();

    for cu_file in &cu_files {
        let cu_path = kernel_dir.join(cu_file);
        println!("cargo:rerun-if-changed={}", cu_path.display());
    }

    for cu_file in &cu_files {
        let cu_path = kernel_dir.join(cu_file);
        if !cu_path.exists() {
            println!("cargo:warning=Kernel file {} not found", cu_file);
            continue;
        }

        let obj_name = cu_file.replace(".cu", ".o");
        let obj_path = out_dir.join(&obj_name);

        let nvcc_path = PathBuf::from(&cuda_path).join("bin").join("nvcc");

        // Compile with nvcc
        let mut cmd = std::process::Command::new(&nvcc_path);
        cmd.args(["-c", "-O3", "--use_fast_math"]);
        cmd.args(&gencode_args);
        cmd.args([
            "-I",
            cuda_include.to_str().unwrap(),
            "-o",
            obj_path.to_str().unwrap(),
            cu_path.to_str().unwrap(),
        ]);
        let status = cmd.status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                println!(
                    "cargo:warning=nvcc failed for {} with exit code {:?}",
                    cu_file,
                    s.code()
                );
            }
            Err(e) => {
                panic!(
                    "CUDA feature was requested, but nvcc could not be executed from {}: {}",
                    nvcc_path.display(),
                    e
                );
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
                return;
            }
        }
    }

    panic!("CUDA feature was requested, but CUDA kernels were not compiled successfully.");
}
