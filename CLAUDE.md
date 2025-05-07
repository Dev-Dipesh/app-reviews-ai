# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands
- Install dependencies: `make install` or `pip install -r requirements.txt`
- Run linting: `make lint` (runs isort and black)
- Run tests: `make test` or `pytest tests`
- Run a specific test: `pytest tests/test_file.py::TestClass::test_function -v`
- Run analysis pipeline: `make run` or `python run.py`
- Run with custom parameters: `python run.py --max-reviews 1000`

## Code Style Guidelines
- **Formatting**: Follow PEP 8 conventions; use black and isort for formatting
- **Imports**: Group imports: stdlib, 3rd-party, local; use absolute imports from src
- **Typing**: Use type hints for function parameters and return values
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Docs**: Include docstrings with parameter descriptions in all functions/classes
- **Modules**: Maintain the modular architecture with proper interfaces
- **Error Handling**: Use try/except with specific exceptions; log errors properly
- **Testing**: Write unit tests for each module in the tests directory

When making changes, maintain the existing modular architecture and follow the interface patterns. Handle errors gracefully with appropriate logging.