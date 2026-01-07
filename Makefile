# TorchRL Development Makefile

.PHONY: clean build develop test

# Clean all build artifacts (use when switching Python/PyTorch versions)
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf torchrl/*.egg-info/
	rm -f torchrl/_torchrl*.so
	rm -f torchrl/version.py
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Build C++ extensions in-place
build:
	python setup.py build_ext --inplace

# Full clean + build
rebuild: clean build

# Development install (editable)
develop: rebuild
	pip install -e . --no-build-isolation

# Run tests
test:
	python -m pytest test/ -v --timeout 60
