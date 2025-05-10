---
description: Python and Machine Learning specific best practices for the project.
globs: ["**/*.py"]
---

# Python & ML Best Practices

## Code Style & Formatting
- Adhere to PEP 8 style guidelines.
- Use a code formatter like Black for consistent formatting (e.g., max line length 88-100 characters).
- Use `isort` or a similar tool to keep imports organized (alphabetical, grouped by type).
- Naming Conventions:
    - `snake_case` for functions, methods, variables, and module names.
    - `PascalCase` (CapWords) for class names.
    - `UPPER_SNAKE_CASE` for constants.
    - Prepend internal/private attributes/methods with a single underscore (`_internal_var`).

## Type Hinting
- Use type hints for all function signatures (arguments and return values) and critical variables.
- Utilize the `typing` module for complex types (`List`, `Dict`, `Tuple`, `Optional`, `Any`, `Callable`, etc.).
- For NumPy arrays, consider using `numpy.typing.NDArray` with dtype information if possible (e.g., `NDArray[np.float64]`).
- For Pandas DataFrames/Series, consider type hinting but acknowledge limitations. Stubs or comments can clarify schemas.

## Virtual Environments
- Always use a virtual environment (e.g., `venv`, `conda`) for project dependencies.
- Include a `requirements.txt` (pinned versions) or `pyproject.toml` with `poetry` or `pdm` for dependency management.
- Keep development and production dependencies separate if applicable.

## Modularity and Structure
- Organize code into logical modules and packages (e.g., `data_processing`, `models`, `utils`, `training`, `evaluation`).
- Use `src/` layout for your main application code (e.g., `src/trader/`).
- Configuration Management:
    - Store configurations in separate files (e.g., YAML, JSON, .env) or use environment variables.
    - Avoid hardcoding paths, API keys, or hyperparameters directly in the code.
    - Consider using libraries like `Hydra` or `Pydantic` for managing complex configurations.

## Data Handling (Pandas/NumPy)
- Optimize Pandas operations: use vectorized operations instead of loops where possible.
- Be mindful of memory usage with large datasets. Load only necessary columns, use appropriate dtypes (e.g., `category` for low-cardinality strings).
- Clearly document DataFrame schemas (column names, dtypes, expected values) where they are passed or returned.

## Machine Learning Workflow
- **Reproducibility**: 
    - Set random seeds (`random.seed()`, `numpy.random.seed()`, `torch.manual_seed()`, etc.) for stochastic processes.
    - Version control your data, code, and models (or model metadata).
    - Log experiment parameters, metrics, and artifacts (e.g., using MLflow, W&B).
- **Model Training**: 
    - Separate training scripts from model definitions.
    - Implement clear logging of training progress (loss, metrics per epoch/batch).
    - Save model checkpoints regularly.
- **Evaluation**: 
    - Use appropriate evaluation metrics for your trading task (e.g., Sharpe ratio, Sortino ratio, max drawdown, accuracy, F1-score for classification aspects).
    - Implement cross-validation or a robust train/validation/test split strategy, especially for time-series data (e.g., walk-forward validation).

## Logging
- Use the `logging` module for application logging instead of `print()` statements for non-interactive parts.
- Configure log levels appropriately (DEBUG, INFO, WARNING, ERROR, CRITICAL).
- Log important events, errors, and state changes.

## Testing for ML
- Unit test data preprocessing functions for expected outputs and edge cases.
- Test feature engineering logic.
- Write integration tests for the end-to-end training/prediction pipeline with small dummy data.
- Test for data leakage between train/validation/test sets. 