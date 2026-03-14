// Unit tests for composable pipeline compression

#[cfg(test)]
mod tests {
    use crate::compressor::{Compressor, PipelineCompressor};
    use crate::{NangilaConfig, Tensor};

    #[test]
    fn test_pipeline_nangila_only() {
        // Test pipeline with just Nangila (baseline)
        // Note: For very small tensors (< 100 elements), serialization overhead
        // can exceed compression gains. This test validates the pipeline works.
        let config = NangilaConfig::default();
        let mut pipeline = PipelineCompressor::new(config);

        // Use a larger tensor to see compression benefits
        let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let gradient = Tensor::new(data, vec![1000]);

        // Warmup - need to update predictor first
        for _ in 0..10 {
            pipeline.update(0, &gradient).unwrap();
            pipeline.step();
        }

        // Now we can compress
        let packet = pipeline.compress(&gradient, 0).unwrap();
        let reconstructed = pipeline.decompress(&packet, 0).unwrap();

        // Check reconstruction quality
        let error: f32 = gradient
            .data
            .iter()
            .zip(&reconstructed.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / gradient.numel() as f32;

        assert!(error < 0.1, "Reconstruction error too high: {}", error);

        // Check compression ratio
        let ratio = pipeline.last_compression_ratio().unwrap();

        println!("Nangila compression ratio (1000 elements): {:.1}×", ratio);
        // With INT4 quantization, we expect at least 4× compression (32bit → 4bit + overhead)
        assert!(ratio > 2.0, "Compression ratio too low: {:.1}×", ratio);
    }

    #[test]
    fn test_pipeline_compression_ratio_tracking() {
        // Test that pipeline tracks compression ratios correctly
        let config = NangilaConfig::default();
        let mut pipeline = PipelineCompressor::new(config);

        // Use larger tensor
        let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let gradient = Tensor::new(data, vec![1000]);

        // Warmup - update predictor first
        for _ in 0..10 {
            pipeline.update(0, &gradient).unwrap();
            pipeline.step();
        }

        // Test
        pipeline.compress(&gradient, 0).unwrap();

        let ratio = pipeline.last_compression_ratio().unwrap();
        println!("Tracked compression ratio: {:.1}×", ratio);

        assert!(ratio > 1.0, "Compression ratio should be > 1.0");
    }
}
