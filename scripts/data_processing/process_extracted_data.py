import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import yaml
import sys
import shutil
from typing import Tuple

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
    from data_utils import clear_directory, detect_anomalies, EXPECTED_DAILY_ENTRIES, PRICE_COLS
except ImportError:
    logging.error("Could not import utility functions from data_utils. Ensure the file exists in the same directory.")
    sys.exit(1)

# Get logger instance
logger = get_logger("ProcessExtractedData")

def process_single_ticker_file(
    input_csv_path: Path,
    output_dir: Path,
    filter_complete_days: bool,
    filter_anomalies: bool,
    anomaly_threshold: float
) -> bool:
    """
    Reads an extracted ticker CSV, applies filters, and saves it to the processed directory if valid.

    Args:
        input_csv_path (Path): Path to the extracted ticker CSV file.
        output_dir (Path): Directory to save the processed CSV file.
        filter_complete_days (bool): If True, filter for EXPECTED_DAILY_ENTRIES.
        filter_anomalies (bool): If True, filter based on anomaly detection.
        anomaly_threshold (float): Threshold for anomaly detection.

    Returns:
        bool: True if the file was processed and saved successfully, False otherwise (skipped or error).
    """
    try:
        df = pd.read_csv(input_csv_path)
        filename = input_csv_path.name

        # --- Essential Columns Check (Basic) ---
        # Check for essential price columns needed for anomaly detection if requested
        if filter_anomalies and not all(col in df.columns for col in PRICE_COLS):
             logger.warning(f"[{filename}] Skipping: Missing required price columns ({PRICE_COLS}) for anomaly check.")
             return False
        # Check for a time column if needed for completeness or other steps
        if filter_complete_days and 'window_start' not in df.columns:
            logger.warning(f"[{filename}] Skipping: Missing 'window_start' for completeness check.")
            return False

        # --- Filter by Completeness ---
        if filter_complete_days:
            if len(df) != EXPECTED_DAILY_ENTRIES:
                logger.debug(f"[{filename}] Skipping: Incomplete data (found {len(df)}, expected {EXPECTED_DAILY_ENTRIES}).")
                return False

        # --- Filter by Anomalies ---
        if filter_anomalies:
            is_anomalous = detect_anomalies(df, anomaly_threshold)
            if is_anomalous:
                logger.debug(f"[{filename}] Skipping: Detected anomaly (threshold: {anomaly_threshold} std dev).")
                return False

        # --- Save Valid File ---
        output_file = output_dir / filename
        # Use copy instead of move to keep extracted data intact if needed elsewhere
        shutil.copy2(input_csv_path, output_file) # copy2 preserves metadata
        logger.debug(f"[{filename}] Passed filters. Saved to processed directory.")
        return True

    except pd.errors.EmptyDataError:
        logger.warning(f"[{input_csv_path.name}] Skipping: Extracted file is empty.")
        return False
    except FileNotFoundError:
        logger.error(f"[{input_csv_path.name}] Skipping: File not found (unexpected). Maybe deleted after listing?")
        return False
    except Exception as e:
        logger.error(f"Error processing file {input_csv_path.name}: {e}")
        return False


def run_processing(
    extracted_dir: str,
    processed_dir: str,
    max_workers: int,
    clear_output: bool,
    filter_complete_days: bool,
    filter_anomalies: bool,
    anomaly_threshold: float,
):
    """
    Finds extracted CSV files and runs the processing/filtering step in parallel.
    """
    extracted_path = Path(extracted_dir)
    processed_path = Path(processed_dir)

    if not extracted_path.exists() or not extracted_path.is_dir():
        logger.error(f"Extracted data directory not found: {extracted_path}")
        return

    if clear_output:
        clear_directory(processed_path)
    else:
        processed_path.mkdir(parents=True, exist_ok=True) # Ensure exists

    # --- File Discovery ---
    # Find all individual ticker CSV files in the extracted directory
    files_to_process = sorted(list(extracted_path.glob('*.csv')))

    if not files_to_process:
        logger.warning(f"No *.csv files found to process in {extracted_path}. Did the extraction step run successfully?")
        return

    logger.info(f"Found {len(files_to_process)} extracted files to process.")

    # --- Parallel Processing ---
    success_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_ticker_file,
                file,
                processed_path,
                filter_complete_days,
                filter_anomalies,
                anomaly_threshold
            ): file
            for file in files_to_process
        }

        processed_files_counter = 0
        for future in futures: # Iterate through completed futures
            extracted_file = futures[future]
            processed_files_counter += 1
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    skipped_count += 1

                if processed_files_counter % 1000 == 0: # Log progress periodically
                     logger.info(f"Processed {processed_files_counter}/{len(files_to_process)} files... (Saved: {success_count}, Skipped: {skipped_count})")

            except Exception as exc:
                 logger.error(f"Error getting result for {extracted_file.name}: {exc}")
                 skipped_count += 1

    logger.info(f"Processing complete. Files saved: {success_count}. Files skipped (filtered or error): {skipped_count}.")
    logger.info(f"Processed data saved to: {processed_path}")


if __name__ == "__main__":
    # --- Logging Setup ---
    log_file = Path("logs") / "process_extracted_data.log"
    setup_logging(
        log_file_path=log_file,
        root_level=logging.INFO,
        level_overrides={
            "ProcessExtractedData": logging.INFO,
            "DataManager": logging.INFO
        }
    )
    # ---------------------

    # --- Configuration Loading ---
    base_dir = Path(__file__).resolve().parent.parent.parent  # Go up one more level to reach project root
    config_path = base_dir / "config" / "process_config.yaml"  # Config file name
    config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
            if config is None: config = {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Create 'config/process_config.yaml'.")
            # Example config structure:
            logger.info("Example `process_config.yaml`:\n"
                        "extracted_dir: data/extracted\n"
                        "processed_dir: data/processed\n"
                        "max_workers: 4\n"
                        "clear_output: true\n"
                        "filter_complete_days: true\n"
                        "filter_anomalies: true\n"
                        "anomaly_std_dev_threshold: 4.0")
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
        extracted_dir_rel = config["extracted_dir"]
        processed_dir_rel = config["processed_dir"]
        max_workers = config["max_workers"]
        clear_output = config.get("clear_output", True)
        filter_complete_days = config.get("filter_complete_days", True)
        filter_anomalies = config.get("filter_anomalies", True)
        anomaly_threshold = config.get("anomaly_std_dev_threshold", 4.0)
    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)
    # ---------------------------

    # --- Path Construction ---
    extracted_dir = base_dir / extracted_dir_rel
    processed_dir = base_dir / processed_dir_rel
    # -------------------------

    logger.info(
        f"Starting extracted data processing. Extracted dir: {extracted_dir}, Processed dir: {processed_dir}"
    )

    run_processing(
        extracted_dir=str(extracted_dir),
        processed_dir=str(processed_dir),
        max_workers=max_workers,
        clear_output=clear_output,
        filter_complete_days=filter_complete_days,
        filter_anomalies=filter_anomalies,
        anomaly_threshold=anomaly_threshold,
    )

    logger.info("Processing script finished.") 