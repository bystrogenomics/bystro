# Assumes you have run ". .initialize-conda-env.sh"; since each make command runs in a separate subshell we need this to happen first

build:
	cd python && maturin build --release && cd ../

develop:
	cd python && maturin develop && cd ../

serve-dev: develop
        ray stop && ray start --head
	pm2 delete all 2> /dev/null || true && pm2 start startup.yml
