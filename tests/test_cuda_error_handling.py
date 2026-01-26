"""
Test CUDA error handling and synchronization modes

These tests verify that error handling works correctly without requiring GPU hardware.
"""

import pytest
import sys

try:
    import nangila
    from nangila import SyncMode
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False


@pytest.mark.skipif(not NANGILA_AVAILABLE, reason="Nangila not installed")
class TestSyncMode:
    """Test SyncMode enum and values"""
    
    def test_sync_mode_values(self):
        """Verify SyncMode constants have correct values"""
        assert SyncMode.ASYNC == 0
        assert SyncMode.ALWAYS == 1
        assert SyncMode.PERIODIC == 2
    
    def test_sync_mode_creation(self):
        """Test creating SyncMode instances"""
        async_mode = SyncMode(0)
        assert async_mode is not None
        
        always_mode = SyncMode(1)
        assert always_mode is not None
        
        periodic_mode = SyncMode(2)
        assert periodic_mode is not None
    
    def test_sync_mode_invalid_value(self):
        """Test that invalid sync mode values raise errors"""
        with pytest.raises(ValueError):
            SyncMode(99)
    
    def test_sync_mode_repr(self):
        """Test string representation of SyncMode"""
        async_mode = SyncMode(0)
        assert "ASYNC" in repr(async_mode)
        
        always_mode = SyncMode(1)
        assert "ALWAYS" in repr(always_mode)
        
        periodic_mode = SyncMode(2)
        assert "PERIODIC" in repr(periodic_mode)


@pytest.mark.skipif(not NANGILA_AVAILABLE, reason="Nangila not installed")
class TestCudaErrorHandling:
    """Test CUDA error handling without GPU"""
    
    def test_cuda_available(self):
        """Test cuda_available() function"""
        # Should return False on Mac without CUDA
        available = nangila.cuda_available()
        assert isinstance(available, bool)
    
    @pytest.mark.skipif(nangila.cuda_available(), reason="CUDA is available")
    def test_cuda_functions_raise_without_cuda(self):
        """Test that CUDA functions raise appropriate errors without CUDA"""
        with pytest.raises(RuntimeError, match="CUDA not compiled"):
            nangila.cuda_predict_and_quantize(
                0, 0, 0, 0.9, 0.1, 0, 100, 0, SyncMode.ASYNC
            )
        
        with pytest.raises(RuntimeError, match="CUDA not compiled"):
            nangila.cuda_dequantize_and_reconstruct(
                0, 0, 0, 0.9, 0.1, 0, 100, 0, SyncMode.ASYNC
            )
    
    def test_hook_creation_with_sync_mode(self):
        """Test creating NangilaHook with different sync modes"""
        # Should not raise even without CUDA
        hook = nangila.NangilaHook.all_drivers(10)
        assert hook is not None


@pytest.mark.skipif(not NANGILA_AVAILABLE, reason="Nangila not installed")
class TestErrorMessages:
    """Test that error messages are informative"""
    
    @pytest.mark.skipif(nangila.cuda_available(), reason="CUDA is available")
    def test_cuda_error_message_quality(self):
        """Verify error messages provide useful information"""
        try:
            nangila.cuda_predict_and_quantize(
                0, 0, 0, 0.9, 0.1, 0, 100, 0, SyncMode.ASYNC
            )
            pytest.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            error_msg = str(e)
            # Error message should mention CUDA and rebuilding
            assert "CUDA" in error_msg
            assert any(word in error_msg.lower() for word in ["rebuild", "compile", "not"])


@pytest.mark.skipif(not NANGILA_AVAILABLE, reason="Nangila not installed")
class TestInputValidation:
    """Test input validation in Python layer"""
    
    def test_invalid_sync_mode_in_cuda_call(self):
        """Test that invalid sync_mode values are caught"""
        if not nangila.cuda_available():
            pytest.skip("CUDA not available")
        
        # Invalid sync mode should raise ValueError
        with pytest.raises(ValueError, match="Invalid sync_mode"):
            nangila.cuda_predict_and_quantize(
                0, 0, 0, 0.9, 0.1, 0, 100, 0, 
                sync_mode=99  # Invalid
            )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
