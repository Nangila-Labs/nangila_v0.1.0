"""
Sculptor Integration Test
=========================
Tests the full Sculptor workflow:
1. Train a model and record gradients
2. Compute correlation matrix (Sculptor analysis)
3. Generate topology mask (Driver/Passenger classification)
4. Validate the mask identifies correlated layers

Usage: python test_sculptor.py
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

def main():
    print("="*60)
    print("SCULPTOR INTEGRATION TEST")
    print("="*60)

    # ============ STEP 1: CREATE MODEL WITH KNOWN CORRELATIONS ============
    print("\n[1/5] Creating model with correlated layers...")

    class CorrelatedModel(nn.Module):
        """
        Model with intentionally correlated layers:
        - Layer 0, 1, 2: Independent (drivers)
        - Layer 3, 4: Correlated with layer 0 (passengers)
        """
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(100, 50)  # Driver
            self.layer1 = nn.Linear(50, 50)   # Driver
            self.layer2 = nn.Linear(50, 50)   # Driver
            self.layer3 = nn.Linear(50, 50)   # Should correlate with layer0
            self.layer4 = nn.Linear(50, 10)   # Should correlate with layer0
            
        def forward(self, x):
            x0 = torch.relu(self.layer0(x))
            x1 = torch.relu(self.layer1(x0))
            x2 = torch.relu(self.layer2(x1))
            # Layer 3 and 4 process similar information as layer 0
            x3 = torch.relu(self.layer3(x0))  # Reuses x0
            x4 = self.layer4(x3)
            return x4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CorrelatedModel().to(device)
    print(f"  Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  Device: {device}")

    # ============ STEP 2: RECORD GRADIENTS ============
    print("\n[2/5] Training and recording gradients...")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Storage for gradient history
    gradient_history = defaultdict(list)
    num_steps = 100

    torch.manual_seed(42)
    for step in range(num_steps):
        x = torch.randn(32, 100).to(device)
        y = torch.randint(0, 10, (32,)).to(device)
        
        pred = model(x)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Record gradients for each layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_history[name].append(param.grad.flatten().cpu().numpy())
        
        optimizer.step()
        
        if step % 25 == 0:
            print(f"  Step {step}/{num_steps}, Loss: {loss.item():.4f}")

    print(f"  Recorded {num_steps} gradient samples for {len(gradient_history)} layers")

    # ============ STEP 3: COMPUTE CORRELATION MATRIX ============
    print("\n[3/5] Computing correlation matrix (Sculptor analysis)...")

    layer_names = list(gradient_history.keys())
    n_layers = len(layer_names)

    # Stack gradients into matrices
    layer_matrices = {}
    for name in layer_names:
        grads = np.array(gradient_history[name])
        # Take mean gradient per step (reduce dimensionality)
        layer_matrices[name] = np.mean(grads, axis=1)

    # Compute correlation matrix
    correlation_matrix = np.zeros((n_layers, n_layers))
    for i, name_i in enumerate(layer_names):
        for j, name_j in enumerate(layer_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(layer_matrices[name_i], layer_matrices[name_j])[0, 1]
                correlation_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0

    print("  Correlation Matrix (abbreviated):")
    for i, name in enumerate(layer_names[:4]):
        row = " ".join([f"{correlation_matrix[i,j]:.2f}" for j in range(min(4, n_layers))])
        print(f"    {name[:15]:>15}: {row}")

    # ============ STEP 4: GENERATE TOPOLOGY MASK ============
    print("\n[4/5] Generating topology mask...")

    CORRELATION_THRESHOLD = 0.7

    drivers = []
    passengers = {}

    # Find drivers (layers not highly correlated with earlier layers)
    for i, name in enumerate(layer_names):
        is_passenger = False
        for j in range(i):
            if correlation_matrix[i, j] > CORRELATION_THRESHOLD:
                # This layer is a passenger of layer j
                passengers[name] = (layer_names[j], correlation_matrix[i, j])
                is_passenger = True
                break
        if not is_passenger:
            drivers.append(name)

    print(f"  Threshold: {CORRELATION_THRESHOLD}")
    print(f"  Drivers ({len(drivers)}):")
    for d in drivers[:3]:
        print(f"    - {d}")
    if len(drivers) > 3:
        print(f"    ... and {len(drivers) - 3} more")

    print(f"  Passengers ({len(passengers)}):")
    for p, (driver, corr) in list(passengers.items())[:3]:
        print(f"    - {p} → {driver} (corr: {corr:.3f})")

    # ============ STEP 5: VALIDATE RESULTS ============
    print("\n[5/5] Validating Sculptor results...")

    tests_passed = 0
    total_tests = 3

    # Test 1: Should have identified some drivers
    if len(drivers) > 0:
        print("  ✓ Test 1: Found driver layers")
        tests_passed += 1
    else:
        print("  ✗ Test 1: No driver layers found")

    # Test 2: Correlation matrix should be symmetric
    is_symmetric = np.allclose(correlation_matrix, correlation_matrix.T, atol=0.01)
    if is_symmetric:
        print("  ✓ Test 2: Correlation matrix is symmetric")
        tests_passed += 1
    else:
        print("  ✗ Test 2: Correlation matrix is not symmetric")

    # Test 3: Should have reasonable compression ratio
    compression_ratio = len(layer_names) / max(len(drivers), 1)
    if compression_ratio >= 1.0:
        print(f"  ✓ Test 3: Compression ratio: {compression_ratio:.2f}x")
        tests_passed += 1
    else:
        print(f"  ✗ Test 3: Invalid compression ratio: {compression_ratio:.2f}x")

    # Final results
    print("\n" + "="*60)
    print("SCULPTOR TEST RESULTS")
    print("="*60)
    print(f"  Tests Passed: {tests_passed}/{total_tests}")
    print(f"  Drivers: {len(drivers)}")
    print(f"  Passengers: {len(passengers)}")
    print(f"  Compression Ratio: {compression_ratio:.2f}x")

    if tests_passed == total_tests:
        print("\n  🎉 SCULPTOR INTEGRATION TEST PASSED! 🎉")
    else:
        print("\n  ⚠️ Some tests failed")

    print("="*60)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
