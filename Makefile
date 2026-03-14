
.PHONY: all install clean test check python-smoke release-check

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

# Run the v0.1 Python smoke baseline in a local virtualenv
python-smoke:
	python3 -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/python -m pip install maturin pytest
	./.venv/bin/maturin develop --release -F python
	./.venv/bin/python -m pytest -q

# Run the documented local release validation flow for v0.1.x
release-check:
	cargo test
	$(MAKE) python-smoke
