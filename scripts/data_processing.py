import pandas as pd
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import datetime
import re
import math
import sys
import yaml
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split

# Use root logger - configuration handled by main script
logger = logging.getLogger("DataProcessing")

# --- Helper Function for Calculating Return ---
def _calculate_return(file_path: Path) -> float | None:
    """Calculate buy-and-hold return (last_close / first_close - 1) for a file."""
    try:
        df = pd.read_csv(file_path)
        if 'close' not in df.columns:
            logger.warning(f"Skipping return calculation: 'close' column not found in {file_path.name}")
            return None
        if len(df) < 2:
            logger.warning(f"Skipping return calculation: Less than 2 data points in {file_path.name}")
            return None

        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]

        if first_close is None or pd.isna(first_close) or first_close == 0:
            logger.warning(f"Skipping return calculation: Invalid first close price ({first_close}) in {file_path.name}")
            return None
        if last_close is None or pd.isna(last_close):
             logger.warning(f"Skipping return calculation: Invalid last close price ({last_close}) in {file_path.name}")
             return None

        return (last_close / first_close) - 1.0
    except pd.errors.EmptyDataError:
        logger.warning(f"Skipping return calculation: File is empty {file_path.name}")
        return None
    except Exception as e:
        logger.error(f"Error calculating return for {file_path.name}: {e}")
        return None

def process_file(input_path, output_dir, filter_complete_days, filter_usd_only):
    """
    Process a single compressed CSV file and create one file per ticker per day.

    Args:
        input_path (Path): Path to the input compressed CSV file
        output_dir (Path): Directory to save the processed data
        filter_complete_days (bool): If True, filter to keep only tickers with exactly 1440 entries for the day
        filter_usd_only (bool): If True, filter to keep only tickers ending with -USD
    """
    try:
        # Read the compressed CSV file
        df = pd.read_csv(input_path, compression="gzip")

        # Filter to only keep USD tickers if requested
        if filter_usd_only and "ticker" in df.columns:
            original_ticker_count = len(df["ticker"].unique())
            df = df[df["ticker"].str.endswith("-USD")]
            usd_ticker_count = len(df["ticker"].unique())

            if usd_ticker_count < original_ticker_count:
                logger.info(
                    f"Filtered out {original_ticker_count - usd_ticker_count} non-USD tickers in {input_path}"
                )

            # If no USD tickers, log a warning and return
            if df.empty:
                logger.warning(
                    f"No USD tickers found in {input_path}. File will be empty."
                )
                return True # Return True indicating process completed (even if output is empty)

        # Process the data - Example operations:
        # 1. Convert timestamp to datetime
        if "window_start" in df.columns:
            df["window_start"] = pd.to_datetime(df["window_start"], unit="ns")

        # 2. Filter out any rows with missing values
        df = df.dropna()

        # 4. Filter for tickers with exactly 1440 entries per day (minute-level data for full day)
        if filter_complete_days and "ticker" in df.columns:
            # Count entries per ticker
            ticker_counts = df["ticker"].value_counts()
            # Keep only tickers with exactly 1440 entries
            complete_day_tickers = ticker_counts[ticker_counts == 1440].index

            # Log how many tickers were filtered out
            orig_ticker_count = len(df["ticker"].unique())
            complete_ticker_count = len(complete_day_tickers)

            if complete_ticker_count < orig_ticker_count:
                logger.info(
                    f"Filtered out {orig_ticker_count - complete_ticker_count} tickers without 1440 entries in {input_path}"
                )

            # Filter the dataframe to keep only those tickers
            df = df[df["ticker"].isin(complete_day_tickers)]

            # If no tickers have 1440 entries, log a warning
            if df.empty:
                logger.warning(
                    f"No tickers with exactly 1440 entries found in {input_path}. File will be empty."
                )
                return True # Return True indicating process completed

        # Dictionary to keep track of processed tickers
        processed_tickers = []
        # Keep track if any file was actually written for this input GZ
        file_written = False

        # Process each ticker and create one file per ticker per day
        for ticker_name, ticker_df in df.groupby("ticker"):
            try:
                # Clean ticker name for filename (replace invalid characters)
                safe_ticker = re.sub(r'[\\/*?:"<>|]', "_", ticker_name)

                # Sort the DataFrame by timestamp
                if "window_start" in ticker_df.columns:
                    ticker_df = ticker_df.sort_values("window_start")

                    # Group by date to create one file per day
                    ticker_df["date"] = ticker_df["window_start"].dt.date

                    # Process each day's data separately
                    for date, day_df in ticker_df.groupby("date"):
                        # Create a file for this ticker and date
                        date_str = date.strftime("%Y-%m-%d")
                        # Replace _X_ with _ in the filename
                        safe_ticker_fixed = safe_ticker.replace("X_", "")
                        output_file = output_dir / f"{date_str}_{safe_ticker_fixed}.csv"

                        # Remove the date column before saving
                        day_df = day_df.drop(columns=["date"])

                        # Save to CSV
                        day_df.to_csv(output_file, index=False)
                        file_written = True # Mark that at least one file was written
                else:
                    # If no timestamp column, use the current date for the filename
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    # Replace _X_ with _ in the filename
                    safe_ticker_fixed = safe_ticker.replace("X_", "")
                    output_file = output_dir / f"{date_str}_{safe_ticker_fixed}.csv"
                    ticker_df.to_csv(output_file, index=False)
                    file_written = True # Mark that at least one file was written

                processed_tickers.append(ticker_name)
            except Exception as e:
                logger.error(
                    f"Error processing data for ticker {ticker_name} from {input_path}: {str(e)}"
                )

        if file_written:
             logger.debug(f"Processed {len(processed_tickers)} tickers from {input_path}")
        # Return True if the function completed without fatal error, False otherwise
        return True

    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False


def clear_directory(directory_path):
    """
    Clear all contents of a directory.

    Args:
        directory_path (Path): Path to the directory to clear
    """
    path = Path(directory_path)

    if path.exists():
        logger.info(f"Clearing directory: {path}")
        # Remove all files and subdirectories
        try:
            shutil.rmtree(path)
            logger.info(f"Directory cleared: {path}")
        except OSError as e:
            logger.error(f"Error clearing directory {path}: {e}")
            # Handle cases where the directory might be in use or permissions are denied
            # Try removing contents individually if rmtree fails
            try:
                for item in path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                logger.info(f"Cleared contents of directory: {path}")
            except Exception as inner_e:
                logger.error(f"Could not clear contents of directory {path}: {inner_e}")
                return  # Exit if clearing fails critically

    # Recreate the empty directory
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not recreate directory {path}: {e}")


def perform_train_val_test_split(
    all_files: List[Path],
    output_dir: Path,
    train_subdir: str,
    val_subdir: str,
    test_subdir: str,
    test_ratio: float,
    validation_size: int,
    seed: int,
):
    """
    Splits the processed CSV files into train, validation, and test sets.
    - Test set is randomly sampled (test_ratio).
    - Validation set (validation_size) is sampled from the remaining files
      based on buy-and-hold return distribution.
    - Training set gets the rest.

    Args:
        all_files (List[Path]): List of all processed CSV file paths.
        output_dir (Path): The base directory where subdirs will be created.
        train_subdir (str): Name for the training subdirectory.
        val_subdir (str): Name for the validation subdirectory.
        test_subdir (str): Name for the testing subdirectory.
        test_ratio (float): Proportion of files for the test set.
        validation_size (int): Number of files for the validation set.
        seed (int): Random seed for reproducibility.
    """
    if not all_files:
        logger.warning("No files provided for splitting.")
        return

    n_total = len(all_files)
    n_test = int(n_total * test_ratio)
    n_initial_train = n_total - n_test

    if n_test <= 0 or n_initial_train <= 0:
        logger.error(f"Cannot split {n_total} files with test_ratio {test_ratio}. Need >0 files for both train and test.")
        return
    if validation_size <= 0:
        logger.error(f"validation_size ({validation_size}) must be positive.")
        return
    if validation_size >= n_initial_train:
        logger.error(f"validation_size ({validation_size}) must be less than the number of initial training files ({n_initial_train}).")
        return

    logger.info(f"Starting split of {n_total} files into Train/Validation/Test.")
    logger.info(f"Test set size: {n_test} ({test_ratio*100:.1f}%)`)")
    logger.info(f"Validation set size: {validation_size} (selected from remaining {n_initial_train})")

    # 1. Create initial Train/Test split
    try:
        initial_train_files, test_files = train_test_split(
            all_files, test_size=test_ratio, random_state=seed
        )
    except Exception as e:
         logger.error(f"Error during initial train_test_split: {e}")
         return

    logger.info(f"Initial split: {len(initial_train_files)} potential train/val files, {len(test_files)} test files.")

    # 2. Calculate returns for the initial training set
    returns_map: List[Tuple[Path, float]] = []
    logger.info("Calculating buy-and-hold returns for potential training files...")
    skipped_count = 0
    for file_path in initial_train_files:
        ret = _calculate_return(file_path)
        if ret is not None:
            returns_map.append((file_path, ret))
        else:
            skipped_count += 1
    if skipped_count > 0:
        logger.warning(f"Skipped return calculation for {skipped_count} files due to errors or invalid data.")

    if not returns_map:
         logger.error("Could not calculate returns for any initial training files. Cannot select validation set.")
         # Fallback: Just put all initial_train_files into train
         train_files = initial_train_files
         validation_files = []
    elif len(returns_map) <= validation_size:
        logger.warning(f"Number of files with valid returns ({len(returns_map)}) is not greater than validation_size ({validation_size}). Using all valid files for validation.")
        validation_files = [item[0] for item in returns_map]
        # Find files that were skipped and put them in train
        valid_files_set = set(validation_files)
        train_files = [f for f in initial_train_files if f not in valid_files_set]
    else:
        # 3. Select Validation files based on return distribution
        logger.info(f"Selecting {validation_size} validation files based on return distribution...")
        # Sort by return value
        returns_map.sort(key=lambda x: x[1])

        # Select indices evenly distributed across the sorted returns
        indices = np.linspace(0, len(returns_map) - 1, validation_size, dtype=int)
        validation_files_set = set(returns_map[i][0] for i in indices)
        validation_files = list(validation_files_set) # Keep unique files

        # Ensure we didn't get fewer than requested due to duplicates if indices weren't unique
        if len(validation_files) < validation_size:
            logger.warning(f"Selected fewer validation files ({len(validation_files)}) than requested ({validation_size}) due to non-unique indices. This is unexpected.")
            # Simple fix: add more from the end until size is met (less ideal distribution)
            needed = validation_size - len(validation_files)
            additional_indices = np.linspace(len(returns_map) - needed, len(returns_map) - 1, needed, dtype=int)
            for i in additional_indices:
                file_to_add = returns_map[i][0]
                if file_to_add not in validation_files_set:
                     validation_files.append(file_to_add)
                     validation_files_set.add(file_to_add)


        # 4. Final Training set is the remainder
        train_files = [
            f for f in initial_train_files if f not in validation_files_set
        ]

        # Log some info about selected validation files' returns
        val_returns = [ret for path, ret in returns_map if path in validation_files_set]
        if val_returns:
            logger.info(f"  Validation file returns - Min: {min(val_returns):.4f}, Max: {max(val_returns):.4f}, Mean: {np.mean(val_returns):.4f}")

    n_train = len(train_files)
    n_val = len(validation_files)
    n_test = len(test_files) # Re-confirm test count
    logger.info(f"Final split sizes: Train={n_train}, Validation={n_val}, Test={n_test}")
    if n_train + n_val + n_test != n_total:
         logger.warning(f"Consistency check failed: Train({n_train}) + Val({n_val}) + Test({n_test}) = {n_train+n_val+n_test} != Total({n_total})")


    # 5. Move files
    train_path = output_dir / train_subdir
    val_path = output_dir / val_subdir
    test_path = output_dir / test_subdir

    # Clear and recreate train/val/test directories
    clear_directory(train_path)
    clear_directory(val_path)
    clear_directory(test_path)

    moved_train, moved_val, moved_test = 0, 0, 0
    errors_moving = 0

    def move_files(file_list, dest_path):
        count = 0
        err_count = 0
        for file_path in file_list:
            try:
                # Check if file still exists before moving (it should)
                if file_path.exists():
                     shutil.move(str(file_path), str(dest_path / file_path.name))
                     count += 1
                else:
                    logger.warning(f"File {file_path} not found for moving to {dest_path}. It might have been moved already or deleted.")
                    err_count += 1 # Count as error if file disappeared
            except Exception as e:
                logger.error(f"Error moving file {file_path} to {dest_path}: {e}")
                err_count += 1
        return count, err_count

    logger.info(f"Moving {n_train} files to {train_path}...")
    moved, errors = move_files(train_files, train_path)
    moved_train += moved
    errors_moving += errors

    logger.info(f"Moving {n_val} files to {val_path}...")
    moved, errors = move_files(validation_files, val_path)
    moved_val += moved
    errors_moving += errors

    logger.info(f"Moving {n_test} files to {test_path}...")
    moved, errors = move_files(test_files, test_path)
    moved_test += moved
    errors_moving += errors

    logger.info(
        f"File moving complete. Moved {moved_train} to Train, {moved_val} to Validation, {moved_test} to Test."
    )
    if errors_moving > 0:
        logger.error(f"{errors_moving} errors occurred during file moving.")


def process_data(
    raw_dir: str,
    output_dir: str,
    max_workers: int,
    filter_complete_days: bool,
    clear_output: bool,
    filter_usd_only: bool,
    year_filter: str | None, # Correct type hint
    perform_split: bool,
    # New split parameters
    test_ratio: float,
    validation_size: int,
    seed: int,
    # Subdir names remain
    train_subdir: str,
    val_subdir: str,
    test_subdir: str,
):
    """
    Process all compressed CSV files from raw directory to CSV format,
    creating one file per ticker per day in a single output directory.
    Then performs the split based on new strategy if requested.

    Args:
        raw_dir (str): Path to the raw data directory
        output_dir (str): Path to save the processed data
        max_workers (int): Maximum number of worker processes
        filter_complete_days (bool): If True, filter to keep only tickers with exactly 1440 entries for the day
        clear_output (bool): If True, clear the output directory before processing
        filter_usd_only (bool): If True, filter to keep only tickers ending with -USD
        year_filter (str | None): If provided, only process files from this year (e.g., '2025')
        perform_split (bool): If True, perform train/val/test split after processing.
        test_ratio (float): Ratio of files for the test set.
        validation_size (int): Number of files for the validation set.
        seed (int): Random seed for split reproducibility.
        train_subdir (str): Subdirectory name for training data.
        val_subdir (str): Subdirectory name for validation data.
        test_subdir (str): Subdirectory name for testing data.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    # --- Start: Clearing and File Discovery (largely unchanged) ---
    # Clear the output directory if requested
    if clear_output:
        # Clear the base output path AND potential subdirs from previous runs
        clear_directory(output_path)
        clear_directory(output_path / train_subdir)
        clear_directory(output_path / val_subdir)
        clear_directory(output_path / test_subdir)
        # Recreate base output dir after clearing
        output_path.mkdir(exist_ok=True)
    else:
        # Just ensure the base directory exists
        output_path.mkdir(exist_ok=True)

    # Gather all compressed CSV files to process
    files_to_process = []
    csv_pattern = re.compile(r".*\.csv\.gz$")

    # If year filter is provided, adjust raw_path
    if year_filter:
        year_dir = raw_path / year_filter
        if year_dir.exists() and year_dir.is_dir():
            logger.info(f"Filtering for files from year: {year_filter}")
            raw_path = year_dir
        else:
            logger.warning(f"Year directory {year_filter} not found in {raw_path}")
            if not (raw_path / year_filter).exists():
                logger.error(
                    f"Directory {raw_path / year_filter} does not exist. Exiting."
                )
                return

    # Check if raw_path itself contains the gz files
    direct_files = [f for f in raw_path.glob('*.csv.gz') if f.is_file()]
    if direct_files:
         files_to_process.extend(direct_files)
    else:
        # Assume year/month structure if no direct files found
        logger.info(f"No direct *.csv.gz files found in {raw_path}, scanning subdirectories (year/month structure assumed)...")
        for item in sorted(raw_path.iterdir()):
            if not item.is_dir(): continue
            if item.name.isdigit() and len(item.name) == 4: # Year directory
                if year_filter and item.name != year_filter: continue
                for month_dir in sorted(item.iterdir()):
                    if not month_dir.is_dir(): continue
                    files_to_process.extend(
                         [f for f in month_dir.glob('*.csv.gz') if f.is_file()]
                    )
            elif item.name.isdigit() and len(item.name) == 2: # Month directory directly under raw_path?
                 files_to_process.extend(
                     [f for f in item.glob('*.csv.gz') if f.is_file()]
                 )

    if not files_to_process:
        logger.error(f"No *.csv.gz files found to process in {raw_path} or its subdirectories.")
        return

    logger.info(f"Found {len(files_to_process)} files to process")
    # --- End: File Discovery ---

    # --- Start: Parallel Processing (unchanged) ---
    success_count = 0
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in files_to_process:
            future = executor.submit(
                process_file, file, output_path, filter_complete_days, filter_usd_only
            )
            futures.append(future)

        for i, future in enumerate(futures):
             try:
                 result = future.result() # Get result or raise exception
                 if result:
                     success_count += 1
                 else:
                     # process_file logs error internally, just count failure
                     failure_count += 1
             except Exception as exc:
                 logger.error(f"Error in processing future {i} (file: {files_to_process[i].name}): {exc}")
                 failure_count += 1

    logger.info(
        f"Processing complete. Success: {success_count}, Failures: {failure_count}"
    )
    logger.info(f"Processed data initially saved to: {output_path}")
    # --- End: Parallel Processing ---

    # --- Start: NEW Splitting Logic ---
    if perform_split:
        logger.info("Gathering list of successfully processed files for splitting...")
        # List all CSV files created in the output directory
        # This assumes process_file saves them directly to output_path
        try:
            processed_files = sorted([f for f in output_path.glob("*.csv") if f.is_file()])
        except Exception as e:
            logger.error(f"Error listing processed files in {output_path} for split: {e}")
            processed_files = []

        if not processed_files:
             logger.warning(f"No processed *.csv files found in {output_path}. Skipping split.")
        else:
            logger.info(f"Found {len(processed_files)} processed files to split.")
            perform_train_val_test_split(
                all_files=processed_files,
                output_dir=output_path,
                train_subdir=train_subdir,
                val_subdir=val_subdir,
                test_subdir=test_subdir,
                test_ratio=test_ratio,
                validation_size=validation_size,
                seed=seed,
            )
    else:
        logger.info("Train/val/test split not requested.")
    # --- End: NEW Splitting Logic ---


# --- UPDATED Main execution block ---
if __name__ == "__main__":
    # --- Logging Setup (unchanged) ---
    try:
        from src.utils.utils import setup_global_logging
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logging.warning("Could not find setup_global_logging, using basic config.")
        setup_global_logging = None

    if setup_global_logging:
        log_file = Path("logs") / "data_processing.log"
        log_file.parent.mkdir(exist_ok=True)
        setup_global_logging(log_file_path=log_file, root_level=logging.INFO)
    # --------------------------------

    # --- Configuration Loading (load new keys) ---
    base_dir = Path(__file__).resolve().parent.parent # Project root
    config_path = base_dir / "config" / "data_processing_config.yaml"
    config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
            if config is None: config = {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Cannot proceed.")
            sys.exit(1)
    except ImportError:
        logger.error("PyYAML library not found. Install with `pip install pyyaml`.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}.")
        sys.exit(1)

    # Get parameters from config (using new keys)
    try:
        raw_data_dir_rel = config["raw_dir"]
        output_dir_rel = config["output_dir"]
        max_workers = config["max_workers"]
        filter_complete_days = config["filter_complete_days"]
        clear_output = config["clear_output"]
        filter_usd_only = config["filter_usd_only"]
        year_filter_yaml = config.get("year_filter", None) # Use get for optional
        perform_split = config["perform_split"]
        # NEW Split parameters
        test_ratio = config["test_ratio"]
        validation_size = config["validation_size"]
        seed = config["seed"]
        # Subdir names
        train_subdir = config["train_subdir"]
        val_subdir = config["val_subdir"]
        test_subdir = config["test_subdir"]
    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)
    # ------------------------------------------

    # --- Path Construction and Validation (unchanged) ---
    raw_data_dir = base_dir / raw_data_dir_rel
    output_dir = base_dir / output_dir_rel
    year_filter = str(year_filter_yaml) if year_filter_yaml is not None else None
    # Basic validation
    if not raw_data_dir.exists() or not raw_data_dir.is_dir():
         logger.error(f"Raw data directory does not exist or is not a directory: {raw_data_dir}")
         sys.exit(1)
    if not (0 < test_ratio < 1):
        logger.error(f"test_ratio ({test_ratio}) must be between 0 and 1.")
        sys.exit(1)
    if not isinstance(validation_size, int) or validation_size <= 0:
        logger.error(f"validation_size ({validation_size}) must be a positive integer.")
        sys.exit(1)
    # -------------------------------------------------

    logger.info(
        f"Running data processing. Raw dir: {raw_data_dir}, Output dir: {output_dir}"
    )

    # --- Call process_data with updated signature ---
    process_data(
        raw_dir=str(raw_data_dir),
        output_dir=str(output_dir),
        max_workers=max_workers,
        filter_complete_days=filter_complete_days,
        clear_output=clear_output,
        filter_usd_only=filter_usd_only,
        year_filter=year_filter,
        perform_split=perform_split,
        # Pass new split parameters
        test_ratio=test_ratio,
        validation_size=validation_size,
        seed=seed,
        # Pass subdir names
        train_subdir=train_subdir,
        val_subdir=val_subdir,
        test_subdir=test_subdir,
    )
    # -----------------------------------------------
