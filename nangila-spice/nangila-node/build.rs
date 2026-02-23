use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cuda/cusparse_solver.cu");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // Link the required NVIDIA libraries
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cusparse");

        // Handle SURF nvhpc module paths dynamically since CXX=g++ overrides the original path
        let surf_base = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.7";
        if std::path::Path::new(surf_base).exists() {
            println!("cargo:rustc-link-search=native={surf_base}/math_libs/12.9/lib64");
            println!("cargo:rustc-link-search=native={surf_base}/cuda/12.9/lib64");
        }

        cc::Build::new()
            .cuda(true)
            .file("cuda/cusparse_solver.cu")
            .compile("cusparse_solver");
    }
}
