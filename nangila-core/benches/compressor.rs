use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nangila_core::{
    compressor::Compressor, compressor::PredictionResidualCompressor, config::NangilaConfig,
    dgc::DGCCompressor, power_sgd::PowerSGDCompressor, Tensor,
};

// 1D tensor for Predictor/DGC
fn make_tensor_1d(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) / 100.0).collect();
    Tensor::new(data, vec![size])
}

// 2D tensor for PowerSGD
fn make_tensor_2d(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) / 100.0).collect();
    let side = (size as f64).sqrt() as usize;
    if side * side == size {
        Tensor::new(data, vec![side, side])
    } else {
        // Fallback to 1xN if not square
        Tensor::new(data, vec![1, size])
    }
}

fn bench_compressors(c: &mut Criterion) {
    let sizes = [10_000, 1_000_000]; // 10K (small), 1M (medium)
                                     // 10M might be too slow for quick micro-bench, stick to these for now.

    let mut group = c.benchmark_group("Compressors");

    for size in sizes {
        // 1. Prediction Residual (Baseline)
        group.bench_with_input(
            BenchmarkId::new("PredictionResidual", size),
            &size,
            |b, &s| {
                let mut config = NangilaConfig::default();
                config.warmup_steps = 0; // Disable warmup delay
                let mut compressor = PredictionResidualCompressor::new(config);
                let tensor = make_tensor_1d(s);
                // Warmup - needs 2 updates for history
                compressor.update(0, &tensor).unwrap();
                compressor.update(0, &tensor).unwrap();

                b.iter(|| compressor.compress(black_box(&tensor), 0).unwrap());
            },
        );

        // 2. DGC (Sparsity 99.9%)
        group.bench_with_input(BenchmarkId::new("DGC_99.9%", size), &size, |b, &s| {
            let mut config = NangilaConfig::default();
            config.dgc_sparsity = 0.999;
            let mut compressor = DGCCompressor::new(config);
            let tensor = make_tensor_1d(s);

            b.iter(|| compressor.compress(black_box(&tensor), 0).unwrap());
        });

        // 3. PowerSGD (Rank 1)
        group.bench_with_input(BenchmarkId::new("PowerSGD_Rank1", size), &size, |b, &s| {
            let mut config = NangilaConfig::default();
            config.power_sgd_rank = 1;
            let mut compressor = PowerSGDCompressor::new(config);
            let tensor = make_tensor_2d(s);

            b.iter(|| compressor.compress(black_box(&tensor), 0).unwrap());
        });

        // 4. PowerSGD (Rank 4)
        group.bench_with_input(BenchmarkId::new("PowerSGD_Rank4", size), &size, |b, &s| {
            let mut config = NangilaConfig::default();
            config.power_sgd_rank = 4;
            let mut compressor = PowerSGDCompressor::new(config);
            let tensor = make_tensor_2d(s);

            b.iter(|| compressor.compress(black_box(&tensor), 0).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compressors);
criterion_main!(benches);
