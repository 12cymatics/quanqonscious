PYTHON ?= python3

.PHONY: all native proofs check

all: native

native:
	$(MAKE) -C native all

proofs:
	./lean/build.sh

check:
	$(PYTHON) lean/test_kernel.py
	$(MAKE) -C native check
	$(PYTHON) -m unittest discover -s tests -p 'test_acceptance.py' -v
