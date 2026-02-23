use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cuda/cusparse_solver.cu");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // Link the required NVIDIA libraries
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cusparse");

        // Handle SURF nvhpc module paths (e.g. /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/math_libs/12.9/lib64)
        if let Ok(cxx) = env::var("CXX") {
            if cxx.contains("hpc_sdk") {
                // cxx is typically: /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvc++
                let parts: Vec<&str> = cxx.split("/compilers/").collect();
                if parts.len() == 2 {
                    let base_path = parts[0];
                    // Link math libs (cusparse)
                    println!("cargo:rustc-link-search=native={base_path}/math_libs/12.9/lib64");
                    println!("cargo:rustc-link-search=native={base_path}/math_libs/12.4/lib64");
                    println!("cargo:rustc-link-search=native={base_path}/math_libs/12.3/lib64");
                    // Link cuda toolkit libs (cudart)
                    println!("cargo:rustc-link-search=native={base_path}/cuda/12.9/lib64");
                    println!("cargo:rustc-link-search=native={base_path}/cuda/12.4/lib64");
                    println!("cargo:rustc-link-search=native={base_path}/cuda/12.3/lib64");
                }
            }
        }

        cc::Build::new()
            .cuda(true)
            .file("cuda/cusparse_solver.cu")
            .compile("cusparse_solver");
    }
}
