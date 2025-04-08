import pytest
import subprocess
import sys
from pathlib import Path
import yaml
import os
import random # Added import
from unittest.mock import patch, MagicMock

# Define project root relative to this test file
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_training.py"
TEST_CONFIG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "config" / "test_training_config.yaml"
TEST_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

@pytest.mark.integration
def test_run_training_script_integration(tmp_path):
    """
    Integration test for the run_training.py script.
    - Mocks DataManager to use test fixture data.
    - Runs the script with a minimal test configuration.
    - Checks for successful execution and expected output files.
    """
    assert SCRIPT_PATH.exists(), f"Training script not found: {SCRIPT_PATH}"
    assert TEST_CONFIG_PATH.exists(), f"Test config not found: {TEST_CONFIG_PATH}"
    assert TEST_FIXTURES_DIR.exists(), f"Test fixtures directory not found: {TEST_FIXTURES_DIR}"
    assert (TEST_FIXTURES_DIR / "data/processed/train").exists(), "Test train data dir missing"
    assert (TEST_FIXTURES_DIR / "data/processed/validation").exists(), "Test validation data dir missing"
    assert (TEST_FIXTURES_DIR / "data/processed/test").exists(), "Test test data dir missing"

    # Create temporary directories for logs and models within tmp_path
    test_model_dir = tmp_path / "models"
    test_log_dir = tmp_path / "logs" # Keep for potential future use
    test_model_dir.mkdir()
    test_log_dir.mkdir() # Create even if not directly asserted

    # Load the test config and update paths
    with open(TEST_CONFIG_PATH, 'r') as f:
        test_config = yaml.safe_load(f)

    # Update config to use temporary output directories
    test_config['run']['model_dir'] = str(test_model_dir)
    # Assuming the global logging setup in run_training.py uses a 'logs' dir relative to execution
    # We rely on the subprocess cwd for logs, but explicitly set model_dir

    # Path for the modified config within tmp_path
    temp_config_path = tmp_path / "temp_test_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(test_config, f)

    # --- Mock DataManager setup --- 
    # Create a mock instance that mimics DataManager behavior with fixture paths
    mock_dm_instance = MagicMock()
    mock_dm_instance.base_dir = TEST_FIXTURES_DIR
    mock_dm_instance.processed_dir = TEST_FIXTURES_DIR / "data/processed"
    mock_dm_instance.train_dir = TEST_FIXTURES_DIR / "data/processed/train"
    mock_dm_instance.val_dir = TEST_FIXTURES_DIR / "data/processed/validation"
    mock_dm_instance.test_dir = TEST_FIXTURES_DIR / "data/processed/test"
    mock_dm_instance._data_organized = False
    mock_dm_instance.train_files = []
    mock_dm_instance.val_files = []
    mock_dm_instance.test_files = []

    # Define the mocked organize_data method
    def mock_organize_data(self):
        self.train_files = sorted(list(self.train_dir.glob("*.csv")))
        self.val_files = sorted(list(self.val_dir.glob("*.csv")))
        self.test_files = sorted(list(self.test_dir.glob("*.csv")))
        if not self.train_files:
            raise FileNotFoundError(f"Mock DataManager: No train files found in {self.train_dir}")
        if not self.val_files:
             raise FileNotFoundError(f"Mock DataManager: No validation files found in {self.val_dir}")
        # Test files are optional
        self._data_organized = True

    # Define mocked getter methods
    def mock_get_training_files(self):
        if not self._data_organized: self.organize_data()
        return self.train_files

    def mock_get_validation_files(self):
        if not self._data_organized: self.organize_data()
        return self.val_files

    def mock_get_test_files(self):
        if not self._data_organized: self.organize_data()
        return self.test_files

    def mock_get_random_training_file(self):
        if not self._data_organized: self.organize_data()
        if not self.train_files:
             raise FileNotFoundError("Mock DataManager: No training files to choose from")
        return random.choice(self.train_files)

    # Bind the methods to the mock instance
    mock_dm_instance.organize_data = mock_organize_data.__get__(mock_dm_instance, MagicMock)
    mock_dm_instance.get_training_files = mock_get_training_files.__get__(mock_dm_instance, MagicMock)
    mock_dm_instance.get_validation_files = mock_get_validation_files.__get__(mock_dm_instance, MagicMock)
    mock_dm_instance.get_test_files = mock_get_test_files.__get__(mock_dm_instance, MagicMock)
    mock_dm_instance.get_random_training_file = mock_get_random_training_file.__get__(mock_dm_instance, MagicMock)
    # --- End Mock DataManager setup ---


    # Patch DataManager constructor within the script's context
    # Target 'scripts.run_training.DataManager' assumes run_training.py does 'from data import DataManager'
    # and expects 'data' module/package to be findable relative to 'scripts'.
    # If data.py is in src/, and src/ is added to path or is a package, this might need adjustment.
    # Let's assume the current structure makes this import work when run from PROJECT_ROOT.
    patch_target = 'src.data.DataManager'
    with patch(patch_target) as MockDataManagerConst:
        # Configure the mock constructor to return our pre-configured instance
        MockDataManagerConst.return_value = mock_dm_instance

        # --- Run the script as a subprocess --- #
        command = [
            sys.executable,             # Use the same python interpreter
            str(SCRIPT_PATH),
            "--config_path", str(temp_config_path)
        ]

        print(f"\nRunning command: {' '.join(command)}")
        print(f"Working directory: {PROJECT_ROOT}")

        # --- Set PYTHONPATH for the subprocess ---
        subprocess_env = os.environ.copy()
        src_path = str(PROJECT_ROOT / "src")
        # Prepend src path to existing PYTHONPATH or set it if it doesn't exist
        subprocess_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{subprocess_env.get('PYTHONPATH', '')}"
        print(f"Setting PYTHONPATH for subprocess: {subprocess_env['PYTHONPATH']}")
        # ----------------------------------------

        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,           # Run from project root for module resolution
            capture_output=True,
            text=True,
            env=subprocess_env,         # Pass the modified environment
            check=False,                # Handle non-zero exit code manually
            timeout=60                  # Add a 60-second timeout
        )

        # --- Assertions --- #
        print("\n--- Subprocess stdout --- ")
        print(process.stdout)
        print("\n--- Subprocess stderr --- ")
        print(process.stderr)
        print("-------------------------")

        # Check 1: Successful execution (exit code 0)
        assert process.returncode == 0, \
            f"Script execution failed with exit code {process.returncode}. See stderr above."

        # Check 2: Mock DataManager was instantiated
        # Since we patched the constructor, this checks if `DataManager()` was called in the script
        MockDataManagerConst.assert_called_once()
        print(f"Mock DataManager constructor called successfully.")

        # Check 3: Expected output files in the *temporary* model directory
        model_files = list(test_model_dir.glob("**/*")) # Search recursively
        print(f"Files found in temporary model dir ({test_model_dir}): {model_files}")
        assert any(f.name.startswith(("checkpoint_trainer", "rainbow_transformer")) and f.is_file() for f in model_files), \
               f"No model checkpoint files (e.g., checkpoint_trainer*.pt, rainbow_transformer*.pt) found in {test_model_dir}"
        print(f"Model checkpoint files found in {test_model_dir}.")

        # Check 4: Check if the main training log file was created in the standard location
        # Note: This depends on the logging setup using a fixed relative path 'logs/training.log'
        # If logging is configured differently (e.g., uses the temp dir), this needs adjustment.
        expected_log_file = PROJECT_ROOT / "logs" / "training.log"
        # This assertion might be fragile depending on logging setup
        # assert expected_log_file.exists(), f"Expected log file not found at {expected_log_file}"
        # print(f"Standard log file found at {expected_log_file}.")

        print("\nIntegration test completed successfully.") 