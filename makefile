
build:
	cargo b

build-em:
	cargo b --features "earth-moon"

run:
	cargo r

run-rk2:
	cargo r -- --rk2

run-em:
	cargo r --features "earth-moon"

run-rk2-em:
	cargo r --features "earth-moon" -- --rk2