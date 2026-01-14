//! Nangila PyTorch DDP Hook
//!
//! This crate provides the DDP communication hook integration for PyTorch.
//! It wraps the core Nangila compression logic and exposes it as a
//! c10d::CommHook compatible interface.
//!
//! # Python Usage
//!
//! ```python
//! import nangila
//!
//! # Calibration phase
//! sculptor = nangila.Sculptor(threshold=0.95)
//! for step in range(calibration_steps):
//!     loss.backward()
//!     for name, param in model.named_parameters():
//!         if param.grad is not None:
//!             sculptor.record(layer_id, param.grad.cpu().numpy())
//!
//! mask_bytes = sculptor.generate_mask()
//! with open("nangila.mask", "wb") as f:
//!     f.write(mask_bytes)
//!
//! # Training phase
//! hook = nangila.NangilaHook.from_mask_file("nangila.mask")
//! # ... use hook for DDP compression
//! ```

pub mod ffi;
pub mod hook;

#[cfg(feature = "python")]
pub mod python;

pub use hook::NangilaHook;


