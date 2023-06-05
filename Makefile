build:
	cd python && maturin build --release && cd ../

develop:
	cd python && maturin develop && cd ../
	pm2 delete all && pm2 start startup.yml

