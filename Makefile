.PHONY: setup install clean lint test run

# Setup virtual environment
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate it with 'source venv/bin/activate' (Linux/Mac) or 'venv\\Scripts\\activate' (Windows)"

# Install dependencies
install:
	pip install -r requirements.txt

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# Run linting
lint:
	isort src
	black src

# Install the package in development mode
dev-install:
	pip install -e .

# Run tests
test:
	pytest tests

# Run the analysis pipeline
run:
	python src/runner.py

# Run with specific parameters
run-custom:
	python src/runner.py --app-id com.goindigo.android --start-date "6 months ago" --max-reviews 5000

# Run the demo notebook
notebook:
	jupyter notebook notebooks/review_analysis_demo.ipynb