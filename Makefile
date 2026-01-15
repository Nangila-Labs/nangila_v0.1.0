
.PHONY: install clean test check

# Default target
all: install

# Install both the main package (Rust/Python) and the C++ DDP extension
install:
	@echo "Installing Nangila Core (Rust/Python)..."
	pip install .
	@echo "Installing Nangila C++ Extension..."
	cd nangila_ddp && pip install .

# Clean build artifacts
clean:
	rm -rf target
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf nangila_ddp/build
	rm -rf nangila_ddp/dist
	rm -rf nangila_ddp/*.egg-info
	rm -rf examples/__pycache__
	rm -rf tests/__pycache__

# Run tests
test:
	pytest tests/

# Check code formatting (if tools available)
check:
	cargo check
