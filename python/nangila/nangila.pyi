# This file is a stub for type checking
# The actual module is implemented in Rust

from typing import Optional, Dict, Any, List
import numpy as np
from numpy.typing import NDArray

__version__: str

def cuda_available() -> bool:
    """Check if CUDA support is available."""
    ...

class NangilaConfig:
    """Configuration for Nangila compression."""
    
    def __init__(
        self,
        momentum: float = 0.9,
        threshold: float = 0.95,
        warmup_steps: int = 1000,
        shadow_run_steps: int = 100,
        quantize_bits: int = 4,
    ) -> None:
        """
        Create a new Nangila configuration.
        
        Args:
            momentum: Momentum coefficient for predictor (default: 0.9)
            threshold: Correlation threshold for Passenger detection (default: 0.95)
            warmup_steps: Steps before enabling compression (default: 1000)
            shadow_run_steps: Steps for predictor to learn dynamics (default: 100)
            quantize_bits: Bit width for quantization (default: 4)
        """
        ...
    
    @staticmethod
    def conservative() -> "NangilaConfig":
        """Create a conservative configuration (safer, less compression)."""
        ...
    
    @staticmethod
    def aggressive() -> "NangilaConfig":
        """Create an aggressive configuration (more compression, needs monitoring)."""
        ...

class Sculptor:
    """
    Offline calibration tool for discovering layer topology.
    
    Run during a calibration phase to identify which layers are
    "Drivers" (transmitted) vs "Passengers" (synthesized locally).
    """
    
    def __init__(self, threshold: float = 0.95) -> None:
        """
        Create a new Sculptor.
        
        Args:
            threshold: Pearson correlation threshold for Passenger detection.
                      Higher = more conservative (fewer Passengers).
        """
        ...
    
    def record(self, layer_id: int, gradient: NDArray[np.float32]) -> None:
        """
        Record a gradient sample for a layer.
        
        Args:
            layer_id: Unique identifier for the layer
            gradient: Flattened gradient values
        """
        ...
    
    def generate_mask(self) -> bytes:
        """
        Generate topology mask from recorded gradients.
        
        Returns:
            Serialized mask bytes that can be saved and loaded.
            
        Raises:
            RuntimeError: If insufficient samples were recorded.
        """
        ...
    
    def num_samples(self) -> int:
        """Number of gradient samples recorded so far."""
        ...
    
    def num_layers(self) -> int:
        """Number of layers being tracked."""
        ...
    
    def reset(self) -> None:
        """Reset all recorded data."""
        ...

class NangilaHook:
    """
    DDP communication hook for gradient compression.
    
    Compresses gradients before All-Reduce and reconstructs them after.
    """
    
    def __init__(
        self,
        mask_bytes: bytes,
        config: Optional[NangilaConfig] = None,
    ) -> None:
        """
        Create a new hook from mask bytes.
        
        Args:
            mask_bytes: Serialized topology mask from Sculptor.generate_mask()
            config: Optional configuration (uses default if None)
        """
        ...
    
    @staticmethod
    def from_mask_file(
        path: str,
        config: Optional[NangilaConfig] = None,
    ) -> "NangilaHook":
        """
        Create a hook by loading mask from a file.
        
        Args:
            path: Path to the mask file
            config: Optional configuration
        """
        ...
    
    @staticmethod
    def all_drivers(num_layers: int) -> "NangilaHook":
        """
        Create a hook where all layers are drivers (no sculpting).
        
        Use this for testing or when no calibration was performed.
        """
        ...
    
    def compress(self, layer_id: int, gradient: NDArray[np.float32]) -> bytes:
        """
        Compress a gradient tensor for transmission.
        
        Args:
            layer_id: Layer identifier
            gradient: Flattened gradient values
            
        Returns:
            Compressed bytes (empty for Passenger layers)
        """
        ...
    
    def decompress(self, layer_id: int, data: bytes) -> NDArray[np.float32]:
        """
        Decompress received data back to gradient.
        
        Args:
            layer_id: Layer identifier  
            data: Compressed bytes from compress()
            
        Returns:
            Reconstructed gradient as NumPy array
        """
        ...
    
    def update(self, layer_id: int, gradient: NDArray[np.float32]) -> None:
        """
        Update predictor state after successful All-Reduce.
        
        IMPORTANT: Call this with the final gradient to maintain
        predictor synchronization across all workers.
        """
        ...
    
    def step(self) -> None:
        """Advance to the next training step."""
        ...
    
    def is_compression_enabled(self) -> bool:
        """Check if compression is currently enabled (after warmup)."""
        ...
    
    def current_step(self) -> int:
        """Get the current training step count."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with keys:
            - step: Current step number
            - compression_enabled: Whether compression is active
            - num_drivers: Number of Driver layers
            - num_passengers: Number of Passenger layers
            - mask_compression_ratio: Compression from topology
            - quantizer_gamma: Current quantization scale
        """
        ...
