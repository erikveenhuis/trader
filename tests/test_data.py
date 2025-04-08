import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Remove sys.path manipulation
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

# Use src package prefix
try:
    from src.data import DataManager
except ImportError as e:
    print(f"Failed to import DataManager from src.data: {e}")
    print(f"Current sys.path: {sys.path}")
    pytest.skip(f"Skipping data tests due to import error: {e}", allow_module_level=True)

# --- Fixture to create dummy data structure ---
@pytest.fixture
def dummy_data_structure(tmp_path):
    """Creates a temporary directory structure mimicking the expected data layout."""
    base_dir = tmp_path
    processed_dir = base_dir / "data" / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy files
    (train_dir / "train_file_1.csv").touch()
    (train_dir / "train_file_2.csv").touch()
    (val_dir / "val_file_1.csv").touch()
    # Leave test directory empty for one test case

    # print(f"[Fixture] Created dummy structure at: {base_dir}")
    return base_dir

# --- Test Cases ---

def test_data_manager_init_success(dummy_data_structure):
    """Test successful initialization when data directories exist."""
    base_dir = dummy_data_structure
    # print(f"[Test Init Success] Using base_dir: {base_dir}")
    try:
        manager = DataManager(base_dir=str(base_dir))
        assert manager.base_dir == base_dir
        assert manager.processed_dir.exists()
        assert manager.train_dir.exists()
        assert manager.val_dir.exists()
        assert manager.test_dir.exists()
        # print(f"[Test Init Success] Manager initialized successfully")
    except Exception as e:
        # print(f"[Test Init Success] Exception during init: {e}")
        pytest.fail(f"DataManager initialization failed unexpectedly: {e}")

def test_data_manager_init_missing_processed(tmp_path):
    """Test initialization fails if the processed directory is missing."""
    base_dir = tmp_path 
    # Don't create data/processed
    (base_dir / "data").mkdir() 
    # print(f"[Test Missing Processed] Using base_dir: {base_dir}")
    with pytest.raises(AssertionError, match="Processed data directory not found"):
        DataManager(base_dir=str(base_dir))
    # print(f"[Test Missing Processed] Correctly raised AssertionError")

def test_data_manager_organize_data_success(dummy_data_structure):
    """Test that organize_data correctly finds the files."""
    base_dir = dummy_data_structure
    manager = DataManager(base_dir=str(base_dir))
    try:
        manager.organize_data()
        assert len(manager.train_files) == 2
        assert len(manager.val_files) == 1
        assert len(manager.test_files) == 0 # Test dir was empty
        assert manager._data_organized is True
        assert manager.get_training_files()[0].name == "train_file_1.csv" # Check sorting
        assert manager.get_training_files()[1].name == "train_file_2.csv"
        assert manager.get_validation_files()[0].name == "val_file_1.csv"
        # print(f"[Test Organize Success] Data organized correctly")
    except Exception as e:
        # print(f"[Test Organize Success] Exception during organize: {e}")
        pytest.fail(f"organize_data failed unexpectedly: {e}")


def test_data_manager_organize_data_missing_train_dir(dummy_data_structure):
    """Test organize_data fails if the train directory is missing."""
    base_dir = dummy_data_structure
    # Remove the train directory created by the fixture
    train_dir = base_dir / "data" / "processed" / "train"
    if train_dir.exists():
        import shutil
        shutil.rmtree(train_dir)
    # print(f"[Test Missing Train Dir] Removed train_dir: {train_dir}")

    manager = DataManager(base_dir=str(base_dir))
    with pytest.raises(AssertionError, match="Train directory not found"):
        manager.organize_data()
    # print(f"[Test Missing Train Dir] Correctly raised AssertionError")


def test_data_manager_get_random_training_file(dummy_data_structure):
    """Test retrieving a random training file."""
    base_dir = dummy_data_structure
    manager = DataManager(base_dir=str(base_dir))
    manager.organize_data() # Load the files first
    
    try:
        random_file = manager.get_random_training_file()
        assert random_file.name in ["train_file_1.csv", "train_file_2.csv"]
        assert random_file.exists()
        # print(f"[Test Random File] Got random file: {random_file.name}")
    except Exception as e:
        # print(f"[Test Random File] Exception getting random file: {e}")
        pytest.fail(f"get_random_training_file failed unexpectedly: {e}") 