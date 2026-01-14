use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nangila_core::{Predictor, Tensor};

fn make_tensor(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    Tensor::new(data, vec![size])
}

fn bench_predictor(c: &mut Criterion) {
    let mut predictor = Predictor::new(0.9, 0);
    
    // Warm up predictor
    for i in 0..10 {
        predictor.update(0, make_tensor(1_000_000));
    }
    predictor.update(0, make_tensor(1_000_000));
    predictor.update(0, make_tensor(1_000_000));

    c.bench_function("predictor_predict_1M", |b| {
        b.iter(|| {
            black_box(predictor.predict(0).unwrap())
        })
    });

    c.bench_function("predictor_update_1M", |b| {
        let tensor = make_tensor(1_000_000);
        b.iter(|| {
            predictor.update(1, black_box(tensor.clone()))
        })
    });
}

criterion_group!(benches, bench_predictor);
criterion_main!(benches);
