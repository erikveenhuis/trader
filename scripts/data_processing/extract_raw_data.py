import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import re
import yaml
import sys
from typing import Optional, Tuple

# Import logging configuration
try:
    from src.utils.logging_config import setup_logging, get_logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.warning("Could not find logging_config, using basic config.")
    get_logger = logging.getLogger

# Import shared functions/constants
try:
    from data_utils import clear_directory  # Changed import to be relative
except ImportError:
    logging.error("Could not import utility functions from data_utils. Ensure the file exists in the same directory.")
    sys.exit(1)

# Get logger instance
logger = get_logger("ExtractRawData")

def extract_gz_and_split_by_ticker(
    input_gz_path: Path,
    output_dir: Path,
    filter_usd_only: bool
) -> Tuple[int, int]:
    """
    Reads a compressed CSV, cleans tickers, optionally filters USD pairs,
    and saves one raw CSV per ticker for that day.

    Args:
        input_gz_path (Path): Path to the input compressed CSV file (.csv.gz).
        output_dir (Path): Directory to save the extracted ticker CSV files.
        filter_usd_only (bool): If True, only process tickers ending in "-USD".

    Returns:
        Tuple[int, int]: Number of tickers successfully saved, number of tickers skipped.
    """
    saved_count = 0
    skipped_count = 0
    try:
        df = pd.read_csv(input_gz_path, compression="gzip")

        if "ticker" not in df.columns or "window_start" not in df.columns:
            logger.warning(f"[{input_gz_path.name}] Skipping: Missing 'ticker' or 'window_start' column.")
            return 0, df['ticker'].nunique() if 'ticker' in df.columns else 0 # Estimate skipped

        # --- Clean Ticker Names ---
        initial_tickers = df["ticker"].unique()
        df["ticker"] = df["ticker"].str.replace(r"^X:", "", regex=True)
        cleaned_tickers = df["ticker"].unique()
        changed_tickers = set(initial_tickers) - set(cleaned_tickers)
        if changed_tickers:
             logger.debug(f"[{input_gz_path.name}] Removed 'X:' prefix from {len(changed_tickers)} tickers.")
        if df["ticker"].isnull().any():
             logger.warning(f"[{input_gz_path.name}] Found null tickers after cleaning. Removing affected rows.")
             df = df.dropna(subset=["ticker"])
             if df.empty:
                  logger.warning(f"[{input_gz_path.name}] Skipping: DataFrame empty after removing null tickers.")
                  return 0, len(initial_tickers) # All original tickers skipped

        # --- Filter USD Tickers (Optional) ---
        if filter_usd_only:
            original_ticker_count = df["ticker"].nunique()
            df = df[df["ticker"].str.endswith("-USD")]
            usd_ticker_count = df["ticker"].nunique()
            skipped_non_usd = original_ticker_count - usd_ticker_count
            if skipped_non_usd > 0:
                logger.debug(f"[{input_gz_path.name}] Filtered out {skipped_non_usd} non-USD tickers.")
                skipped_count += skipped_non_usd

            if df.empty:
                logger.info(f"[{input_gz_path.name}] Skipping: No USD tickers remaining after filtering.")
                return 0, skipped_count

        # --- Convert Timestamp ---
        df["window_start"] = pd.to_datetime(df["window_start"], unit="ns")

        # --- Group by Ticker and Save ---
        for ticker_name, ticker_df in df.groupby("ticker"):
            if ticker_df.empty:
                continue # Should not happen after dropna, but safety check

            try:
                # Determine date from the first timestamp
                file_date = ticker_df["window_start"].iloc[0].date()
                date_str = file_date.strftime("%Y-%m-%d")

                # Clean ticker name for filename
                safe_ticker = re.sub(r'[\/*?"<>|]', "_", ticker_name)

                output_file = output_dir / f"{date_str}_{safe_ticker}.csv"

                # Save raw data for this ticker on this day
                ticker_df.sort_values("window_start").to_csv(output_file, index=False)
                saved_count += 1
                logger.debug(f"[{input_gz_path.name}] Saved extracted file: {output_file.name}")

            except Exception as e:
                logger.error(f"[{input_gz_path.name}] Error saving ticker {ticker_name}: {e}")
                skipped_count += 1

        return saved_count, skipped_count

    except pd.errors.EmptyDataError:
        logger.warning(f"[{input_gz_path.name}] Skipping: File is empty.")
        return 0, 0 # No tickers to save or skip
    except FileNotFoundError:
        logger.error(f"[{input_gz_path.name}] Skipping: File not found.")
        return 0, 0
    except Exception as e:
        logger.error(f"Error processing GZ file {input_gz_path.name}: {e}")
        # Cannot easily determine skipped count here, return 0,0
        return 0, 0


def run_extraction(
    raw_dir: str,
    extracted_dir: str,
    max_workers: int,
    clear_output: bool,
    filter_usd_only: bool,
    year_filter: Optional[str],
):
    """
    Finds raw .csv.gz files and runs the extraction process in parallel.
    """
    raw_path = Path(raw_dir)
    extracted_path = Path(extracted_dir)

    if clear_output:
        clear_directory(extracted_path)
    else:
        extracted_path.mkdir(parents=True, exist_ok=True) # Ensure exists

    # --- File Discovery ---
    files_to_process = []
    search_path = raw_path
    if year_filter:
        year_dir = raw_path / year_filter
        if year_dir.exists() and year_dir.is_dir():
            logger.info(f"Filtering for files from year: {year_filter}")
            search_path = year_dir
        else:
            logger.warning(f"Year directory {year_filter} not found in {raw_path}, searching base raw directory.")
            # Optionally exit if strict year filtering is required:
            # logger.error(f"Year directory {year_dir} does not exist. Exiting.")
            # return

    logger.info(f"Searching for *.csv.gz files in {search_path} and its subdirectories...")
    files_to_process = sorted(list(search_path.rglob('*.csv.gz'))) # Use rglob for recursive search

    if not files_to_process:
        logger.error(f"No *.csv.gz files found in {search_path} or subdirectories.")
        return

    logger.info(f"Found {len(files_to_process)} raw files to extract.")

    # --- Parallel Extraction ---
    total_saved = 0
    total_skipped = 0
    processed_files_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_gz_and_split_by_ticker, file, extracted_path, filter_usd_only): file
            for file in files_to_process
        }

        for future in futures: # Iterate through completed futures
            gz_file = futures[future]
            processed_files_count += 1
            try:
                saved, skipped = future.result()
                total_saved += saved
                total_skipped += skipped
                if saved > 0 or skipped > 0:
                     logger.info(f"Processed GZ {processed_files_count}/{len(files_to_process)}: {gz_file.name} -> Saved: {saved}, Skipped Tickers: {skipped}")
                else:
                     logger.warning(f"Processed GZ {processed_files_count}/{len(files_to_process)}: {gz_file.name} -> No tickers saved or skipped (likely error or empty file).")

            except Exception as exc:
                 logger.error(f"Error getting result for {gz_file.name}: {exc}")
                 # Mark as if it skipped all potential tickers, though we don't know how many
                 total_skipped += 1 # Count the file itself as skipped


    logger.info(f"Extraction complete. Total ticker files saved: {total_saved}. Total tickers skipped: {total_skipped}.")
    logger.info(f"Extracted data saved to: {extracted_path}")


if __name__ == "__main__":
    # --- Logging Setup ---
    log_file = Path("logs") / "extract_raw_data.log"
    setup_logging(
        log_file_path=log_file,
        root_level=logging.INFO,
        level_overrides={
            "ExtractRawData": logging.INFO,
            "DataManager": logging.INFO
        }
    )
    # ---------------------

    # --- Configuration Loading ---
    base_dir = Path(__file__).resolve().parent.parent.parent  # Go up one more level to reach project root
    config_path = base_dir / "config" / "extract_config.yaml"  # New config file name
    config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
            if config is None: config = {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Create 'config/extract_config.yaml'.")
            # Example config structure:
            logger.info("Example `extract_config.yaml`:\n"
                        "raw_dir: data/raw\n"
                        "extracted_dir: data/extracted\n"
                        "max_workers: 4\n"
                        "clear_output: true\n"
                        "filter_usd_only: true\n"
                        "year_filter: null # or e.g., '2023'")
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

    try:
        raw_dir_rel = config["raw_dir"]
        extracted_dir_rel = config["extracted_dir"]
        max_workers = config["max_workers"]
        clear_output = config.get("clear_output", True) # Default to True
        filter_usd_only = config.get("filter_usd_only", True) # Default to True
        year_filter_yaml = config.get("year_filter", None)
    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)
    # ---------------------------

    # --- Path Construction and Validation ---
    raw_data_dir = base_dir / raw_dir_rel
    extracted_dir = base_dir / extracted_dir_rel
    year_filter = str(year_filter_yaml) if year_filter_yaml is not None else None

    if not raw_data_dir.exists() or not raw_data_dir.is_dir():
         logger.error(f"Raw data directory does not exist or is not a directory: {raw_data_dir}")
         sys.exit(1)
    # ------------------------------------

    logger.info(
        f"Starting raw data extraction. Raw dir: {raw_data_dir}, Extracted dir: {extracted_dir}"
    )

    run_extraction(
        raw_dir=str(raw_data_dir),
        extracted_dir=str(extracted_dir),
        max_workers=max_workers,
        clear_output=clear_output,
        filter_usd_only=filter_usd_only,
        year_filter=year_filter,
    )

    logger.info("Extraction script finished.") 