import os
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import logging
from datetime import datetime, timedelta
import random

# Configure logging # REMOVED
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger('DataManager')

class DataManager:
    """Manages data splits for training, validation, and testing."""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Get the absolute path to the workspace
            resolved_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            resolved_base_dir = base_dir

        self.base_dir = Path(resolved_base_dir)
        assert self.base_dir.exists(), f"Base directory not found: {self.base_dir}"
        assert self.base_dir.is_dir(), f"Base path is not a directory: {self.base_dir}"
        
        self.processed_dir = self.base_dir / "data/processed"
        assert self.processed_dir.exists(), f"Processed data directory not found: {self.processed_dir}"
        assert self.processed_dir.is_dir(), f"Processed data path is not a directory: {self.processed_dir}"
        
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "validation"
        self.test_dir = self.processed_dir / "test"
        
        self.train_files = []
        self.val_files = []
        self.test_files = []
        self._data_organized = False # Flag to track if data has been loaded
        
        logger.info(f"Using base directory: {self.base_dir}")
        logger.info(f"Using processed directory: {self.processed_dir}")
        logger.info(f"Expecting data in: {self.train_dir}, {self.val_dir}, {self.test_dir}")
        
        # Data organization is now lazy-loaded via getter methods
    
    # Removed get_file_date method as sorting/splitting is external
    # Removed split_data_by_time method as splitting is external

    def organize_data(self):
        """Load file lists from pre-split train/validation/test directories."""
        if self._data_organized:
             return # Avoid re-organizing

        logger.info("Loading file lists from pre-split directories...")
        
        # Check if directories exist
        assert self.train_dir.is_dir(), f"Train directory not found or is not a directory: {self.train_dir}"
        assert self.val_dir.is_dir(), f"Validation directory not found or is not a directory: {self.val_dir}"
        assert self.test_dir.is_dir(), f"Test directory not found or is not a directory: {self.test_dir}"

        try:
            # Load and sort files from each directory by filename (assumes YYYY-MM-DD format)
            self.train_files = sorted(list(self.train_dir.glob("*.csv")))
            self.val_files = sorted(list(self.val_dir.glob("*.csv")))
            self.test_files = sorted(list(self.test_dir.glob("*.csv")))
            
            # Assert that files were actually found if the directories exist
            assert len(self.train_files) > 0, f"No training CSV files found in {self.train_dir}"
            assert len(self.val_files) > 0, f"No validation CSV files found in {self.val_dir}"
            # Test files might be optional, so only warn if missing
            if len(self.test_files) == 0:
                 logger.warning(f"No test CSV files found in {self.test_dir}")
            
            logger.info(f"Found {len(self.train_files)} training files.")
            logger.info(f"Found {len(self.val_files)} validation files.")
            logger.info(f"Found {len(self.test_files)} test files.")
            
            self._data_organized = True # Mark as organized
            
        except Exception as e:
            logger.error(f"Error organizing data from subdirectories: {e}", exc_info=True)
            self.train_files, self.val_files, self.test_files = [], [], []
            self._data_organized = False 
            raise # Re-raise the exception
    
    def _ensure_organized(self):
        """Internal helper to ensure data is organized before accessing files."""
        if not self._data_organized:
            self.organize_data()
        assert self._data_organized, "Data organization failed, cannot retrieve files."
            
    def get_training_files(self) -> List[Path]:
        """Get training files."""
        self._ensure_organized()
        assert len(self.train_files) > 0, "No training files loaded."
        return self.train_files
    
    def get_validation_files(self) -> List[Path]:
        """Get validation files."""
        self._ensure_organized()
        assert len(self.val_files) > 0, "No validation files loaded."
        return self.val_files
    
    def get_test_files(self) -> List[Path]:
        """Get test files."""
        self._ensure_organized()
        # No assertion here, test files might be empty
        return self.test_files
    
    def get_random_training_file(self) -> Path:
        """Get a random training file."""
        self._ensure_organized()
        assert len(self.train_files) > 0, "Cannot get random file: No training files available."
        random_file = random.choice(self.train_files)
        assert random_file.exists(), f"Chosen random training file does not exist: {random_file}"
        return random_file 