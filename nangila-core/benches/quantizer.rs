use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use nangila_core::{Quantizer, Tensor};

fn make_tensor(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| ((i as f32).sin() * 0.1)).collect();
    Tensor::new(data, vec![size])
}

fn bench_quantizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantizer");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let tensor = make_tensor(size);
        let bytes = size * 4; // FP32 input

        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_function(format!("quantize_{}", size), |b| {
            let mut quantizer = Quantizer::int4();
            b.iter(|| black_box(quantizer.quantize(&tensor, 0, 0)))
        });

        let mut quantizer = Quantizer::int4();
        let compressed = quantizer.quantize(&tensor, 0, 0);

        group.bench_function(format!("dequantize_{}", size), |b| {
            b.iter(|| black_box(quantizer.dequantize(&compressed)))
        });
    }

    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut quantizer = Quantizer::int4();
    let tensor = make_tensor(1_000_000);

    c.bench_function("full_roundtrip_1M", |b| {
        b.iter(|| {
            let compressed = quantizer.quantize(black_box(&tensor), 0, 0);
            black_box(quantizer.dequantize(&compressed))
        })
    });
}

criterion_group!(benches, bench_quantizer, bench_compression_ratio);
criterion_main!(benches);
