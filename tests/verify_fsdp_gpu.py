
import torch
import torch.distributed as dist
import nangila
from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook, NangilaConfig

def test_gpu_native_fsdp_hook():
    """
    Test the GPU-native FSDP hook with mocked distributed group.
    Assumes single GPU available.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return
        
    if not nangila.cuda_available():
        print("Skipping test: Nangila CUDA extension not compiled")
        return

    device = torch.device("cuda:0")
    
    # Mock Process Group (Single Rank or Fake Multi-Rank)
    class MockProcessGroup:
        pass
        
    world_size = 2
    rank = 0
    
    # Patch distributed functions
    original_get_rank = dist.get_rank
    original_get_world_size = dist.get_world_size
    original_all_gather = dist.all_gather
    
    dist.get_rank = lambda group=None: rank
    dist.get_world_size = lambda group=None: world_size
    
    def mock_all_gather(tensor_list, tensor, group=None):
        # Simulate all gather: everyone sends the same tensor (logic simplifiction)
        # OR better: simulate Rank 1 sending something else.
        for i in range(len(tensor_list)):
            if i == rank:
                tensor_list[i].copy_(tensor)
            else:
                # Rank 1 sends slightly different data to test averaging
                # We need to construct a valid compressed tensor for Rank 1?
                # Hard w/o running prediction logic for Rank 1.
                # Just copy Rank 0's data for symmetry.
                tensor_list[i].copy_(tensor)
    
    dist.all_gather = mock_all_gather
    
    try:
        config = NangilaConfig.aggressive()
        pg = MockProcessGroup()
        state = NangilaFSDPState(pg, config)
        
        # Create a "FlatParam" gradient
        grad = torch.randn(1024, device=device, dtype=torch.float32)
        grad_original = grad.clone()
        
        # Run Hook - Step 1
        # Predictor is empty (0).
        # Pred = 0. Residual = grad.
        # Quantized = grad / gamma.
        # Rec = Quantized * gamma.
        # Average = Rec (since both ranks symmetric).
        output = nangila_fsdp_hook(state, grad)
        
        print(f"Step 1 Output Shape: {output.shape}")
        # Expect shard size: 1024 / 2 = 512
        assert output.shape[0] == 512
        
        # Check reconstruction error (quantization noise)
        full_recon = torch.cat([output, output], dim=0) # Symmetric
        error = (grad_original - full_recon).abs().mean()
        rel_error = error / grad_original.abs().mean()
        print(f"Step 1 Reconstruction MAE: {error.item():.6f}, Rel: {rel_error.item():.4f}")
        # First step has no prediction history, so error is higher (full residual)
        # INT4 with range [-8,7] at ~0.1 gamma gives ~0.8 max error
        assert rel_error < 1.0, f"Rel error too high: {rel_error}"
        
        # Run Hook - Step 2 (Predictor Active)
        # New grad correlated with old
        grad2 = grad_original + 0.1 # Small shift
        grad2_orig = grad2.clone()
        
        output2 = nangila_fsdp_hook(state, grad2)
        
        print(f"Step 2 Output Shape: {output2.shape}")
        # Add basic assertion for output2
        assert output2.shape[0] == 512
        
    finally:
        # Restore mocked functions
        dist.get_rank = original_get_rank
        dist.get_world_size = original_get_world_size
        dist.all_gather = original_all_gather
        print("Test teardown complete.")

if __name__ == "__main__":
    test_gpu_native_fsdp_hook()
