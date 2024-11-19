build-python:
	(cd python && maturin build --release)

build-python-dev:
	(cd python && maturin develop)

install-python: build-python
	@WHEEL_FILE=$$(ls python/target/wheels/*cp311-cp311-manylinux*x86_64.whl | head -n 1) && \
	if [ -z "$$WHEEL_FILE" ]; then \
		echo "No manylinux x86_64 wheel found for installation."; \
		exit 1; \
	else \
		echo "Installing $$WHEEL_FILE..."; \
		pip install "$$WHEEL_FILE"; \
	fi

install: install-python

uninstall:
	pip uninstall -y bystro
	binary_path=$(which bystro-vcf 2>/dev/null) && [ -n "$binary_path" ] && rm "$binary_path"

develop: build-python-dev