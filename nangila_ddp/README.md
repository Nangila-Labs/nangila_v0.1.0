# Nangila C++ DDP Hook

Native C++ implementation of the DDP communication hook for production deployments.

## Why C++?

The Python DDP hook incurs ~38 seconds/step overhead due to Python object serialization (pickle). 
The C++ hook bypasses this entirely, achieving near-native throughput (~100ms/step).

## Building

```bash
# Build Rust static library first
cargo build --release -p nangila-hook

# Build C++ extension
cd nangila_ddp
pip install .
```

## Usage

```python
from nangila_ddp_cpp import NangilaDDPHook
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Create hook
hook = NangilaDDPHook(num_layers=1000, warmup_steps=20)

# Register with DDP model
model = DDP(model.cuda())
model.register_comm_hook(state=None, hook=hook)

# Training loop
for step in range(num_steps):
    loss = model(batch).mean()
    loss.backward()
    optimizer.step()
    hook.step()  # Important: advance hook state
```

## Requirements

- PyTorch >= 2.0 with C++ extensions
- CUDA toolkit
- Rust toolchain
