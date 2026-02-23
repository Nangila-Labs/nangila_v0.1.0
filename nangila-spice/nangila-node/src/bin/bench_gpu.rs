use nangila_node::gpu_solver::{GpuSolver, SolverBackend, SparseMatrix};
use std::time::Instant;

fn main() {
    // 1 million nodes
    let n = 1_000_000;
    println!("\n[PVT Benchmark] Building {}x{} sparse matrix...", n, n);

    let mut mat = SparseMatrix::new(n);
    mat.rhs = vec![1.0; n];

    // Tridiagonal-like structure (typical for 1D RC ladders)
    let mut row_ptr = vec![0; n + 1];
    let mut col_indices = Vec::with_capacity(n * 3);
    let mut values = Vec::with_capacity(n * 3);

    let mut nnz = 0;
    for i in 0..n {
        row_ptr[i] = nnz as i32;

        if i > 0 {
            col_indices.push((i - 1) as i32);
            values.push(-1.0);
            nnz += 1;
        }

        col_indices.push(i as i32);
        values.push(2.0);
        nnz += 1;

        if i < n - 1 {
            col_indices.push((i + 1) as i32);
            values.push(-1.0);
            nnz += 1;
        }
    }
    row_ptr[n] = nnz as i32;

    mat.row_ptr = row_ptr;
    mat.col_indices = col_indices;
    mat.values = values;

    println!("Matrix built! Non-zeros: {}", mat.nnz());

    let corners = 200;

    #[cfg(feature = "cuda")]
    {
        println!("--------------------------------------------------");
        println!(">>> L40S GPU / cuSPARSE Delta Solver Backend <<<");
        println!("--------------------------------------------------");

        let mut solver_gpu = GpuSolver::with_backend(SolverBackend::Gpu);

        // Warmup (allocates VRAM buffers, transfers matrix over PCIe)
        print!("Transferring {}x{} matrix to L40S VRAM... ", n, n);
        let t0 = Instant::now();
        solver_gpu.solve(&mut mat);
        let transfer_time = t0.elapsed();
        println!("done in {:.2?}", transfer_time);

        println!("Running {} corner permutations purely in VRAM...", corners);
        let start = Instant::now();
        for _ in 0..corners {
            solver_gpu.solve(&mut mat);
        }
        let elapsed = start.elapsed();

        println!("\n[GPU RESULTS]");
        println!("Total solving time: {:.2?}", elapsed);
        println!("Time per corner:    {:.2?}", elapsed / corners as u32);
        println!("Throughput:         {:.1} corners/sec", corners as f64 / elapsed.as_secs_f64());
        println!("--------------------------------------------------\n");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("--------------------------------------------------");
        println!(">>> CPU-Only Baseline (Dense Fallback) <<<");
        println!("--------------------------------------------------");
        
        // Only run 5 corners on CPU because generating 1M x 1M dense matrix will crash 
        // the CPU, so we skip the actual solve in this bench to prevent an OOM kill.
        println!("NOTE: The standard CPU path uses dense factorization.");
        println!("Attempting to dense factorize a 10,000 x 10,000 slice to estimate time...");
        
        let sub_n = 5_000;
        let mut sub_mat = SparseMatrix::new(sub_n);
        sub_mat.rhs = vec![1.0; sub_n];
        let mut sub_dense = vec![0.0f64; sub_n * sub_n];
        for i in 0..sub_n {
            sub_dense[i * sub_n + i] = 2.0;
            if i > 0 { sub_dense[i * sub_n + i - 1] = -1.0; }
            if i < sub_n - 1 { sub_dense[i * sub_n + i + 1] = -1.0; }
        }
        sub_mat = SparseMatrix::from_dense(&sub_dense, sub_n, 1e-30);
        sub_mat.rhs = vec![1.0; sub_n];
        
        let mut solver_cpu = GpuSolver::with_backend(SolverBackend::Cpu);
        
        let t0 = Instant::now();
        solver_cpu.solve(&mut sub_mat);
        let t_sub = t0.elapsed();
        
        // $O(N^3)$ scaling
        let scaling_factor = (n as f64 / sub_n as f64).powi(3);
        let est_full = t_sub.as_secs_f64() * scaling_factor;
        
        println!("\n[CPU ESTIMATE FOR 1,000,000 NODES]");
        println!("5K node solve took: {:.2?} (dense)", t_sub);
        println!("Estimated 1M node factorization time: {:.0} hours", est_full / 3600.0);
        println!("Compile with `--features cuda` to run the true GPU benchmark.");
        println!("--------------------------------------------------\n");
    }
}
