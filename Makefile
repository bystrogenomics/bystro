ray-start-local:
	ray stop && ray start --head --disable-usage-stats

ray-stop-local:
	ray stop

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

install-go:
	(cd ./go && go install bystro/cmd/dosage)
	(cd ./go && go install bystro/cmd/dosage-filter)
	(cd ./go && go install bystro/cmd/opensearch)

install: install-python install-go

uninstall:
	pip uninstall -y bystro
	(cd ./go && go clean -i bystro/cmd/opensearch)

develop: install-go build-python-dev ray-start-local

run-local: install ray-start-local

pm2:
	pm2 delete all 2> /dev/null || true && pm2 start startup.yml

# Currently assumes that Perl package has been separately installed
serve-local: ray-stop-local run-local pm2

# Currently assumes that Perl package has been separately installed
serve-dev: ray-stop-local develop pm2
