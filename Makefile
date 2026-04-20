PYTHON = venv/Scripts/python.exe
CONFIG = configs/default.yaml

.PHONY: run-lk run-farneback run-raft run-segmentation run-comparison \
        benchmark test format docker-build

run-lk:
	$(PYTHON) run_flow.py --config $(CONFIG) --mode single --method lucas_kanade

run-farneback:
	$(PYTHON) run_flow.py --config $(CONFIG) --mode single --method farneback

run-raft:
	$(PYTHON) run_flow.py --config $(CONFIG) --mode single --method raft

run-segmentation:
	$(PYTHON) run_flow.py --config $(CONFIG) --mode segmentation

run-comparison:
	$(PYTHON) run_flow.py --config $(CONFIG) --mode comparison

benchmark:
	$(PYTHON) run_benchmark.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest tests/ -v

format:
	$(PYTHON) -m black src/ tests/ run_flow.py run_benchmark.py

docker-build:
	docker build -t optical-flow-pipeline .
