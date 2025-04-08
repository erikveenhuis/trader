import pandas as pd
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import datetime
import re
from datetime import datetime
import math # Added for split calculation
import sys # Import sys

# Use root logger - configuration handled by main script
logger = logging.getLogger('DataProcessing')

# Define candle sizes

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
        df = pd.read_csv(input_path, compression='gzip')
        
        # Filter to only keep USD tickers if requested
        if filter_usd_only and 'ticker' in df.columns:
            original_ticker_count = len(df['ticker'].unique())
            df = df[df['ticker'].str.endswith('-USD')]
            usd_ticker_count = len(df['ticker'].unique())
            
            if usd_ticker_count < original_ticker_count:
                logger.info(f"Filtered out {original_ticker_count - usd_ticker_count} non-USD tickers in {input_path}")
            
            # If no USD tickers, log a warning and return
            if df.empty:
                logger.warning(f"No USD tickers found in {input_path}. File will be empty.")
                return True
        
        # Process the data - Example operations:
        # 1. Convert timestamp to datetime
        if 'window_start' in df.columns:
            df['window_start'] = pd.to_datetime(df['window_start'], unit='ns')
        
        # 2. Filter out any rows with missing values
        df = df.dropna()
        
        # 4. Filter for tickers with exactly 1440 entries per day (minute-level data for full day)
        if filter_complete_days and 'ticker' in df.columns:
            # Count entries per ticker
            ticker_counts = df['ticker'].value_counts()
            # Keep only tickers with exactly 1440 entries
            complete_day_tickers = ticker_counts[ticker_counts == 1440].index
            
            # Log how many tickers were filtered out
            orig_ticker_count = len(df['ticker'].unique())
            complete_ticker_count = len(complete_day_tickers)
            
            if complete_ticker_count < orig_ticker_count:
                logger.info(f"Filtered out {orig_ticker_count - complete_ticker_count} tickers without 1440 entries in {input_path}")
            
            # Filter the dataframe to keep only those tickers
            df = df[df['ticker'].isin(complete_day_tickers)]
            
            # If no tickers have 1440 entries, log a warning
            if df.empty:
                logger.warning(f"No tickers with exactly 1440 entries found in {input_path}. File will be empty.")
                return True
        
        # Dictionary to keep track of processed tickers
        processed_tickers = []
        
        # Process each ticker and create one file per ticker per day
        for ticker_name, ticker_df in df.groupby('ticker'):
            try:
                # Clean ticker name for filename (replace invalid characters)
                safe_ticker = re.sub(r'[\\/*?:"<>|]', '_', ticker_name)
                
                # Sort the DataFrame by timestamp
                if 'window_start' in ticker_df.columns:
                    ticker_df = ticker_df.sort_values('window_start')
                    
                    # Group by date to create one file per day
                    ticker_df['date'] = ticker_df['window_start'].dt.date
                    
                    # Process each day's data separately
                    for date, day_df in ticker_df.groupby('date'):
                        # Create a file for this ticker and date
                        date_str = date.strftime('%Y-%m-%d')
                        # Replace _X_ with _ in the filename
                        safe_ticker_fixed = safe_ticker.replace('X_', '')
                        output_file = output_dir / f"{date_str}_{safe_ticker_fixed}.csv"
                        
                        # Remove the date column before saving
                        day_df = day_df.drop(columns=['date'])
                        
                        # Save to CSV
                        day_df.to_csv(output_file, index=False)
                else:
                    # If no timestamp column, use the current date for the filename
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    # Replace _X_ with _ in the filename
                    safe_ticker_fixed = safe_ticker.replace('X_', '')
                    output_file = output_dir / f"{date_str}_{safe_ticker_fixed}.csv"
                    ticker_df.to_csv(output_file, index=False)
                
                processed_tickers.append(ticker_name)
            except Exception as e:
                logger.error(f"Error processing data for ticker {ticker_name} from {input_path}: {str(e)}")
        
        logger.info(f"Processed {len(processed_tickers)} tickers from {input_path}")
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
                return # Exit if clearing fails critically

    # Recreate the empty directory
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not recreate directory {path}: {e}")

def perform_train_val_test_split(output_dir: Path, train_subdir: str, val_subdir: str, test_subdir: str, train_ratio: float, val_ratio: float):
    """
    Splits the processed CSV files in output_dir into train, validation, and test subdirectories based on time.

    Args:
        output_dir (Path): The directory containing the processed CSV files (YYYY-MM-DD_ticker.csv).
        train_subdir (str): The name of the subdirectory for training data.
        val_subdir (str): The name of the subdirectory for validation data.
        test_subdir (str): The name of the subdirectory for testing data.
        train_ratio (float): The proportion of data (by time) to allocate to the training set.
        val_ratio (float): The proportion of data (by time) to allocate to the validation set.
                           The test set gets the remainder.
    """
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or not (0 < train_ratio + val_ratio < 1):
        logger.error(f"Invalid ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. Must be between 0 and 1, and sum must be less than 1.")
        return
        
    logger.info(f"Starting train/validation/test split with train_ratio={train_ratio}, val_ratio={val_ratio}...")
    
    train_path = output_dir / train_subdir
    val_path = output_dir / val_subdir
    test_path = output_dir / test_subdir

    # Clear and recreate train/val/test directories
    clear_directory(train_path)
    clear_directory(val_path)
    clear_directory(test_path)
    
    # List all processed CSV files
    try:
        processed_files = sorted([f for f in output_dir.glob('*.csv') if f.is_file()])
    except Exception as e:
        logger.error(f"Error listing files in {output_dir}: {e}")
        return
        
    if not processed_files:
        logger.warning(f"No processed CSV files found in {output_dir} to split.")
        return

    # Extract unique dates from filenames
    dates = set()
    date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})_.*\.csv$')
    for f in processed_files:
        match = date_pattern.match(f.name)
        if match:
            dates.add(match.group(1))
            
    if not dates:
        logger.warning(f"Could not extract dates from filenames in {output_dir}. Cannot perform split.")
        return

    sorted_dates = sorted(list(dates))
    
    # Determine split dates
    n_dates = len(sorted_dates)
    train_end_index = math.ceil(n_dates * train_ratio)
    val_end_index = math.ceil(n_dates * (train_ratio + val_ratio))

    # Ensure indices are within bounds and handle edge cases
    if train_end_index >= n_dates:
        train_end_index = n_dates -1 # Leave at least one for val/test if possible
    if val_end_index >= n_dates:
         val_end_index = n_dates # If val ratio pushes it to the end, test gets nothing
    if train_end_index < 0: train_end_index = 0
    if val_end_index <= train_end_index: # Ensure val index is after train index if possible
        val_end_index = train_end_index + 1
        if val_end_index >= n_dates:
             val_end_index = n_dates # Adjust if pushing val index goes out of bounds


    train_end_date_str = sorted_dates[train_end_index - 1] if train_end_index > 0 else "" # Date is inclusive for train
    # Validation starts *after* train_end_date_str
    # Validation ends at val_end_date_str (inclusive)
    val_end_date_str = sorted_dates[val_end_index - 1] if val_end_index > 0 and val_end_index <= n_dates else sorted_dates[-1]

    logger.info(f"Split points: Train <= {train_end_date_str}, Validation <= {val_end_date_str}, Test > {val_end_date_str}")
    if train_end_index == 0:
         logger.warning("Train ratio resulted in potentially zero training files.")
    if val_end_index <= train_end_index:
         logger.warning("Validation ratio resulted in potentially zero validation files.")
    if val_end_index >= n_dates:
        logger.warning("Train + Validation ratios cover all data. Test set will be empty.")


    # Move files to train, validation or test directory
    moved_train = 0
    moved_val = 0
    moved_test = 0
    errors_moving = 0
    
    for file_path in processed_files:
        match = date_pattern.match(file_path.name)
        if match:
            file_date_str = match.group(1)
            try:
                if train_end_date_str and file_date_str <= train_end_date_str:
                    shutil.move(str(file_path), str(train_path / file_path.name))
                    moved_train += 1
                elif val_end_date_str and file_date_str <= val_end_date_str:
                     shutil.move(str(file_path), str(val_path / file_path.name))
                     moved_val += 1
                else:
                    shutil.move(str(file_path), str(test_path / file_path.name))
                    moved_test += 1
            except Exception as e:
                logger.error(f"Error moving file {file_path}: {e}")
                errors_moving += 1
        else:
             logger.warning(f"Could not parse date from filename {file_path.name}, skipping move.")


    logger.info(f"Train/val/test split complete. Moved {moved_train} to {train_path}, {moved_val} to {val_path}, {moved_test} to {test_path}.")
    if errors_moving > 0:
        logger.error(f"{errors_moving} errors occurred during file moving.")

def process_data(raw_dir, output_dir, max_workers, filter_complete_days, clear_output, 
                filter_usd_only, year_filter,
                perform_split=True, train_ratio=0.9, val_ratio=0.05,
                train_subdir="train", val_subdir="validation", test_subdir="test"):
    """
    Process all compressed CSV files from raw directory to CSV format,
    creating one file per ticker per day in a single output directory.
    
    Args:
        raw_dir (str): Path to the raw data directory
        output_dir (str): Path to save the processed data
        max_workers (int): Maximum number of worker processes
        filter_complete_days (bool): If True, filter to keep only tickers with exactly 1440 entries for the day
        clear_output (bool): If True, clear the output directory before processing
        filter_usd_only (bool): If True, filter to keep only tickers ending with -USD
        year_filter (str): If provided, only process files from this year (e.g., '2025')
        perform_split (bool): If True, perform train/val/test split after processing.
        train_ratio (float): Ratio of data for training (by time).
        val_ratio (float): Ratio of data for validation (by time).
        train_subdir (str): Subdirectory name for training data.
        val_subdir (str): Subdirectory name for validation data.
        test_subdir (str): Subdirectory name for testing data.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # Clear the output directory if requested
    if clear_output:
        clear_directory(output_path)
    else:
        # Just ensure the directory exists
        output_path.mkdir(exist_ok=True)
    
    # Gather all compressed CSV files to process
    files_to_process = []
    success_count = 0
    failure_count = 0
    
    # Check if the raw_path is a direct directory containing CSV files
    # or a root directory with year/month subdirectories
    csv_pattern = re.compile(r'.*\.csv\.gz$')
    
    # If year filter is provided, check if that year directory exists
    if year_filter:
        year_dir = raw_path / year_filter
        if year_dir.exists() and year_dir.is_dir():
            logger.info(f"Filtering for files from year: {year_filter}")
            raw_path = year_dir
        else:
            logger.warning(f"Year directory {year_filter} not found in {raw_path}")
            if not (raw_path / year_filter).exists():
                logger.error(f"Directory {raw_path / year_filter} does not exist. Exiting.")
                return
    
    if any(csv_pattern.match(f.name) for f in raw_path.iterdir() if f.is_file()):
        # This is a directory containing CSV files directly
        for file in sorted(raw_path.iterdir()):
            if file.is_file() and csv_pattern.match(file.name):
                # If year filter is provided, check if the file matches the year pattern
                if year_filter and not file.name.startswith(year_filter):
                    # Check if the filename contains the year (e.g., 2025-03-28.csv.gz)
                    if not re.search(rf'{year_filter}-\d\d-\d\d', file.name):
                        continue
                files_to_process.append(file)
    else:
        # This is a root directory with year/month structure, or just a year directory
        # Walk through the directory structure to find CSV files
        for item in sorted(raw_path.iterdir()):
            if not item.is_dir():
                continue
                
            # Check if this is a year directory
            if item.name.isdigit() and len(item.name) == 4:
                # If year filter is provided, skip other years
                if year_filter and item.name != year_filter:
                    continue
                    
                # This is a year directory, look for month directories
                for month_dir in sorted(item.iterdir()):
                    if not month_dir.is_dir():
                        continue
                        
                    # Look for CSV files in the month directory
                    for file in sorted(month_dir.iterdir()):
                        if file.is_file() and csv_pattern.match(file.name):
                            files_to_process.append(file)
    
    if not files_to_process:
        logger.error(f"No CSV files found to process in {raw_path}")
        return
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in files_to_process:
            future = executor.submit(
                process_file,
                file,
                output_path,
                filter_complete_days,
                filter_usd_only
            )
            futures.append(future)
        
        # Wait for all processes to complete and count successes/failures
        for future in futures:
            if future.result():
                success_count += 1
            else:
                failure_count += 1
    
    # Log final statistics
    logger.info(f"Processing complete. Success: {success_count}, Failures: {failure_count}")
    logger.info(f"Processed data saved to: {output_path}")

    # Perform train/test split if requested
    if perform_split:
        perform_train_val_test_split(output_path, train_subdir, val_subdir, test_subdir, train_ratio, val_ratio)
    else:
        logger.info("Train/val/test split not requested.")

# Main execution block
if __name__ == "__main__":
    # Configure logging ONLY if run as main script
    # Assumes utils.py exists relative to project root
    try:
        # Attempt relative import suitable for when run as module (-m)
        from ..src.utils.utils import setup_global_logging 
    except ImportError:
        # Fallback if run directly and src is in PYTHONPATH or similar
        try:
             from src.utils.utils import setup_global_logging
        except ImportError:
            # Basic config if setup function not found
            logging.basicConfig(level=logging.INFO, 
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                handlers=[logging.StreamHandler(sys.stdout)])
            logging.warning("Could not find setup_global_logging, using basic config.")
            setup_global_logging = None # Mark as unavailable

    if setup_global_logging:
        log_file = Path("logs") / "data_processing.log"
        log_file.parent.mkdir(exist_ok=True)
        setup_global_logging(log_file_path=log_file, root_level=logging.INFO)
    
    # Set path relative to project root when run directly or as module
    base_dir = Path(__file__).resolve().parent.parent # Assumes script is in /scripts
    raw_data_dir = base_dir / "data" / "raw" # Correct path relative to project root
    output_dir = base_dir / "data" / "processed"
    
    logger.info(f"Running data processing. Raw dir: {raw_data_dir}, Output dir: {output_dir}")
    
    process_data(
        raw_dir=str(raw_data_dir),
        output_dir=str(output_dir),
        max_workers=24,
        filter_complete_days=True,
        clear_output=True,
        filter_usd_only=True,
        year_filter=None,  # Set to specific year (e.g., '2025') to process only that year
        perform_split=True, # Enable the train/val/test split
        train_ratio=0.9,    # 90% train
        val_ratio=0.025,     # 5% validation (remaining 5% is test)
        train_subdir="train", # Subdirectory for training data
        val_subdir="validation",# Subdirectory for validation data
        test_subdir="test"    # Subdirectory for testing data
    ) 