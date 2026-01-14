"""
CIFAR-10 Convergence Test
=========================
Trains a CNN on CIFAR-10 using 2-GPU DDP.
Proves that distributed training converges correctly.

Usage: torchrun --nproc_per_node=2 test_cifar10.py

Expected results: >60% accuracy after 5 epochs
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("="*60)
        print("CIFAR-10 CONVERGENCE TEST")
        print(f"Training on {world_size} GPUs")
        print("="*60)
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    # Distributed data loading
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(trainset, batch_size=256, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    # Simple CNN model
    if rank == 0:
        print("Creating CNN model...")
    
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        nn.Linear(256, 10)
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    if rank == 0:
        print("Training for 5 epochs...")
    
    start_time = time.time()
    
    for epoch in range(5):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        if rank == 0:
            ddp_model.eval()
            correct = total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(rank), targets.to(rank)
                    outputs = ddp_model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/5 | Accuracy: {acc:.2f}%")
        
        dist.barrier()
    
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"\n  Training Time: {elapsed:.1f}s")
        print(f"  Final Accuracy: {acc:.2f}%")
        
        if acc > 60:
            print(f"\n  ✓ CONVERGENCE TEST PASSED!")
            print(f"    DDP training is working correctly")
        else:
            print(f"\n  ✗ CONVERGENCE TEST FAILED")
            print(f"    Expected >60% accuracy")
        print("="*60)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
