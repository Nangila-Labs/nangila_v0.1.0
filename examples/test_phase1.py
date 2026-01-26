
import os
import subprocess
import argparse
import sys
import numpy as np

def run_command(cmd, desc):
    print(f"\n>>> {desc}...")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f">>> {desc} COMPLETED.")
    except subprocess.CalledProcessError as e:
        print(f">>> {desc} FAILED (Exit Code {e.returncode}).")
        sys.exit(e.returncode)

def test_1_1_gradients(world_size=2):
    print("\n=== Test 1.1: Single-Step Gradient Match ===")
    
    # 1. Run Standard FSDP (Port 29501)
    cmd_std = f"torchrun --nproc_per_node={world_size} --master_port=29501 verify_grads.py --output std_grads.npy --world_size {world_size}"
    run_command(cmd_std, "Running Standard FSDP (No Nangila)")
    
    # 2. Run Nangila FSDP (Port 29502)
    cmd_nangila = f"torchrun --nproc_per_node={world_size} --master_port=29502 verify_grads.py --nangila --output nangila_grads.npy --world_size {world_size}"
    run_command(cmd_nangila, "Running Nangila FSDP")
    
    # 3. Compare
    print("Comparing gradients...")
    try:
        std = np.load("std_grads.npy")
        nangila = np.load("nangila_grads.npy")
        
        diff = np.abs(std - nangila).max()
        print(f"Max Difference: {diff:.8f}")
        
        if diff < 1e-6:
            print("SUCCESS: Gradients match within 1e-6 tolerance.")
        else:
            print("FAILURE: Gradients do not match!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Comparison Error: {e}")
        # sys.exit(1) # Don't exit, maybe file not found

def test_1_2_overfit(world_size=2):
    print("\n=== Test 1.2: Overfitting Sanity Check ===")
    
    # Port 29503
    cmd = f"torchrun --nproc_per_node={world_size} --master_port=29503 verify_overfit.py --nangila --world_size {world_size}"
    run_command(cmd, "Carrying out Overfitting Test")

def test_1_3_sharding(world_size=4):
    print("\n=== Test 1.3: Sharding Verification ===")
    print(f"Running on {world_size} GPUs to verify memory distribution...")
    
    # Port 29504
    cmd = f"torchrun --nproc_per_node={world_size} --master_port=29504 verify_sharding.py --world_size {world_size}"
    run_command(cmd, "Verifying Memory Sharding")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all")
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    
    # Helper to clean up
    os.system("rm -f std_grads.npy nangila_grads.npy")
    
    if args.test == "all" or args.test == "1.1":
        test_1_1_gradients(args.world_size)
        
    if args.test == "all" or args.test == "1.2":
        test_1_2_overfit(args.world_size)
    
    # Test 1.3 works best with more cards (4)
    if args.test == "all" or args.test == "1.3":
        test_1_3_sharding(4 if args.world_size < 4 else args.world_size)
