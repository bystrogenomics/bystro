# Assumes you have run ". .initialize-conda-env.sh"; since each make command runs in a separate subshell we need this to happen first

build:
	cd python && maturin build --release && cd ../

develop:
	cd python && maturin develop && cd ../

# Ray must be started with make serve-dev
# without ray start, make serve-dev will succeed, but the handlers that rely on Ray will fail to start
serve-dev: develop
	ray stop && ray start --head
	pm2 delete all 2> /dev/null || true && pm2 start startup.yml

clean:
	@find . -iname "*.bak" -print0 | xargs -0 rm -f
	@rm -f perl/Bytro-0.001.tar.gz
	@rm -rf perl/Bytro-0.001

build-bystro-docker:
	@cd perl && dzil build && docker build -t bystro .
