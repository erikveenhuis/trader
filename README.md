# Trading Agent Project

[Add a brief, high-level description of the project goals and the type of agent being trained here.]

## Project Structure

*   `src/`: Core Python library code (environment, agent, model, trainer, etc.)
*   `scripts/`: Standalone Python scripts (data processing, training runner, analysis).
*   `tests/`: Unit and integration tests.
*   `config/`: Configuration files (e.g., training hyperparameters).
*   `data/`: Project data.
    *   `data/raw/`: Raw input data.
    *   `data/processed/`: Processed data ready for training/evaluation.
*   `logs/`: Log files (ignored by git).
*   `models/`: Saved model checkpoints (ignored by git).
*   `results/`: Output results like plots (ignored by git).
*   `requirements.txt`: Project dependencies.
*   `.gitignore`: Files and directories ignored by git.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd trader
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Data Processing

The data processing script transforms data from `data/raw` to `data/processed`, including splitting into train/validation/test sets.

```bash
PYTHONPATH=src python3 -m scripts.data_processing
```
*(Note: Check `scripts/data_processing.py` for command-line arguments to customize behavior, e.g., year filtering, clearing output directory.)*

## Running Training

To run the training process using the configuration specified in `config/training_config.yaml`:

```bash
PYTHONPATH=src python3 -m scripts.run_training --config_path config/training_config.yaml
```
*   Logs will be generated in `training.log` and `validation.log`.
*   Model checkpoints will be saved in the `models/` directory.
*   Training can be resumed if checkpoints exist (controlled via `resume: true` in the `run` section of the config - *Note: This section might need to be added to the config file*).

## Running Evaluation

To evaluate a trained model:
1.  Ensure the configuration file (`config/training_config.yaml`) has a `run` section specifying `mode: eval` and the `eval_model_prefix` pointing to the saved model (e.g., `models/rainbow_transformer_best`).
2.  Run the training script:
    ```bash
    PYTHONPATH=src python3 -m scripts.run_training --config_path config/training_config.yaml
    ```

## Running Tests

To run the test suite (unit and integration tests) using pytest:

```bash
pytest
``` 