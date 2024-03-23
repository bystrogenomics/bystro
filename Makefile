ray-start-local:
	ray stop && ray start --head --disable-usage-stats

ray-stop-local:
	ray stop

build-python:
	(cd python && maturin build --release)

build-python-dev:
	(cd python && maturin develop)

install-python: build-python
	pip install python/target/wheels/*.whl

install-go:
	(cd ./go && go install bystro/cmd/opensearch)

install: install-python install-go

uninstall:
	pip uninstall -y bystro
	(cd ./go && go clean -i bystro/cmd/opensearch)

develop: install-go build-python-dev ray-start-local

run-local: install ray-start-local

# Currently assumes that Perl package has been separately installed
serve-local: ray-stop-local run-local
	pm2 delete all 2> /dev/null || true && pm2 start startup.yml

# Currently assumes that Perl package has been separately installed
serve-dev: ray-stop-local start-local develop pm2