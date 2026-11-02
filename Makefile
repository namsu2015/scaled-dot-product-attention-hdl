.PHONY: all vectors sim verify test clean

IVERILOG ?= iverilog
VVP ?= vvp

all: test

vectors:
	python3 scripts/golden_model.py --write-vectors

sim: vectors
	mkdir -p build results
	$(IVERILOG) -g2012 -Wall -s tb_attention -o build/attention_sim rtl/attention_core.sv tb/tb_attention.sv
	$(VVP) build/attention_sim

verify: sim
	python3 scripts/golden_model.py --check results/output.txt
	test -s results/attention.vcd
	test -s results/output.txt

test: verify

clean:
	rm -rf build results
