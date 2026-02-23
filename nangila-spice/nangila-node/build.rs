use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cuda/cusparse_solver.cu");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cusparse");

        cc::Build::new()
            .cuda(true)
            .file("cuda/cusparse_solver.cu")
            .compile("cusparse_solver");
    }
}
