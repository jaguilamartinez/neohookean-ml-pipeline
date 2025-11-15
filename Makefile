.PHONY: help setup train-baseline train-pann notebook clean clean-all

DATA_PATH ?= ../neohookean-data-generator/data/consolidated_all.npz

help:
	@echo "Available targets:"
	@echo "  setup         - Create environment"
	@echo "  train-baseline - Train baseline model"
	@echo "  train-pann    - Train PANN model"
	@echo "  notebook      - Launch Jupyter"
	@echo "  clean         - Remove cache"
	@echo "  clean-all     - Full cleanup"

setup:
	python3 -m venv keras_env
	./keras_env/bin/pip install -q --upgrade pip
	./keras_env/bin/pip install -q -r requirements.txt
	@echo "Done. Activate: source keras_env/bin/activate"

train-baseline:
	@test -f $(DATA_PATH) || (echo "Error: $(DATA_PATH) not found" && exit 1)
	python3 -c "from src.models.base_model import run_training; run_training('$(DATA_PATH)')"

train-pann:
	@test -f $(DATA_PATH) || (echo "Error: $(DATA_PATH) not found" && exit 1)
	python3 -c "from src.models.pann_model import run_pann_training; run_pann_training('$(DATA_PATH)')"

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf keras_env output/run_*
