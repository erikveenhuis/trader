import pytest
import pandas as pd
import tempfile
import os
import sys

# Remove sys.path manipulation
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(project_root))

# Use updated import path
try:
    from src.env.trading_env import TradingEnv
except ImportError as e:
    print(f"Failed to import TradingEnv from src.env: {e}")
    print(f"sys.path: {sys.path}")
    raise

# Constants for testing
WINDOW_SIZE = 10
N_FEATURES = 5
INITIAL_BALANCE = 10000.0
TRANSACTION_FEE = 0.001


# Helper function to create mock data CSV
def create_mock_csv(data_dict, temp_dir_path):
    """Creates a mock CSV file in a temporary directory."""
    df = pd.DataFrame(data_dict)
    path = os.path.join(temp_dir_path, "mock_data.csv")
    df.to_csv(path, index=False)
    return path


# Fixture to create a dummy CSV file path
@pytest.fixture
def dummy_csv_path():
    # ... (fixture code) ...
    pass  # Add pass


@pytest.mark.unittest
class TestTradingEnvInitializationErrors:
    """Tests for errors during TradingEnv initialization."""

    def setup_method(self, method):
        # Renamed from __init__
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window_size = WINDOW_SIZE
        self.initial_balance = INITIAL_BALANCE
        self.transaction_fee = TRANSACTION_FEE

    # Remove the __call__ method if it exists
    # def __call__(self):
    #     return self

    def teardown_method(self, method):
        # Renamed from tearDown
        self.temp_dir.cleanup()

    def test_non_existent_data_file(self):
        """Test initializing with a non-existent data file path."""
        non_existent_path = "path/to/non_existent/file.csv"
        with pytest.raises(FileNotFoundError):
            TradingEnv(
                data_path=non_existent_path,
                window_size=self.window_size,
                initial_balance=self.initial_balance,
                transaction_fee=self.transaction_fee,
            )

    def test_missing_column(self):
        """Test initializing with data missing a required column."""
        mock_data_dict = {
            "open": [100, 101, 102, 103, 104, 105, 106],
            "high": [105, 106, 107, 108, 109, 110, 111],
            "low": [95, 96, 97, 98, 99, 100, 101],
            # 'close': [100, 101, 102, 103, 104, 105, 106], # Missing close
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        with pytest.raises(
            AssertionError, match=r"Input data missing.*required columns"
        ):
            TradingEnv(
                data_path=mock_path,
                window_size=self.window_size,
                initial_balance=self.initial_balance,
                transaction_fee=self.transaction_fee,
            )

    def test_non_numeric_data(self):
        """Test initializing with non-numeric data in a required column."""
        mock_data_dict = {
            "open": [100, 101, 102, 103, 104, 105, 106],
            "high": [105, 106, 107, 108, 109, 110, 111],
            "low": [95, 96, 97, 98, 99, 100, 101],
            "close": [100, 101, "invalid", 103, 104, 105, 106],  # Non-numeric
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        # This might raise during pandas read_csv or during the numeric check assertion
        with pytest.raises(AssertionError) as cm:
            TradingEnv(
                data_path=mock_path,
                window_size=self.window_size,
                initial_balance=self.initial_balance,
                transaction_fee=self.transaction_fee,
            )
        assert "Column 'close' must be numeric." in str(cm.value)

    def test_insufficient_rows(self):
        """Test initializing with fewer rows than window_size."""
        num_rows = self.window_size - 1
        mock_data_dict = {
            "open": [100 + i for i in range(num_rows)],
            "high": [105 + i for i in range(num_rows)],
            "low": [95 + i for i in range(num_rows)],
            "close": [100 + i for i in range(num_rows)],
            "volume": [1000] * num_rows,
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        # Expect assertion failure from __init__ (before normalization)
        with pytest.raises(
            AssertionError, match=r"Data length \(\d+\) must be >= window_size \(\d+\)."
        ):
            TradingEnv(
                data_path=mock_path,
                window_size=self.window_size,
                initial_balance=self.initial_balance,
                transaction_fee=self.transaction_fee,
            )

    def test_invalid_window_size(self):
        """Test initializing with invalid window_size."""
        mock_path = create_mock_csv(
            {
                "open": [1] * 10,
                "high": [1] * 10,
                "low": [1] * 10,
                "close": [1] * 10,
                "volume": [1] * 10,
            },
            self.temp_dir.name,
        )
        with pytest.raises(
            AssertionError, match=r"window_size must be an integer >= 1"
        ):
            TradingEnv(data_path=mock_path, window_size=0)
        with pytest.raises(
            AssertionError, match=r"window_size must be an integer >= 1"
        ):
            TradingEnv(data_path=mock_path, window_size=-1)
        with pytest.raises(AssertionError):  # Non-integer
            TradingEnv(data_path=mock_path, window_size=5.5)

    def test_invalid_initial_balance(self):
        """Test initializing with invalid initial_balance."""
        mock_path = create_mock_csv(
            {
                "open": [1] * 10,
                "high": [1] * 10,
                "low": [1] * 10,
                "close": [1] * 10,
                "volume": [1] * 10,
            },
            self.temp_dir.name,
        )
        with pytest.raises(
            AssertionError, match=r"initial_balance must be a non-negative number"
        ):
            TradingEnv(data_path=mock_path, window_size=5, initial_balance=-100)

    def test_invalid_transaction_fee(self):
        """Test initializing with invalid transaction_fee."""
        mock_path = create_mock_csv(
            {
                "open": [1] * 10,
                "high": [1] * 10,
                "low": [1] * 10,
                "close": [1] * 10,
                "volume": [1] * 10,
            },
            self.temp_dir.name,
        )
        with pytest.raises(
            AssertionError, match=r"transaction_fee must be a number between 0.*and 1"
        ):
            TradingEnv(data_path=mock_path, window_size=5, transaction_fee=-0.001)
        with pytest.raises(
            AssertionError, match=r"transaction_fee must be a number between 0.*and 1"
        ):
            TradingEnv(
                data_path=mock_path, window_size=5, transaction_fee=1.0
            )  # Fee must be < 1
