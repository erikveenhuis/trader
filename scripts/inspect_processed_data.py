import logging
from data import DataManager

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger("OrganizeData")


def main():
    # Initialize the data manager
    data_manager = DataManager()

    # Organize the data (this will split files into train/val/test sets)
    data_manager.organize_data()

    # Get the splits
    train_files = data_manager.get_training_files()
    val_files = data_manager.get_validation_files()
    test_files = data_manager.get_test_files()

    # Check if we have any files
    if not train_files and not val_files and not test_files:
        print("\nNo files found in the processed data directory.")
        print("Please ensure there are CSV files in the data/train, data/val, and data/test directories.")
        return

    # Print some example files from each split
    print("\nExample files from each split:")

    if train_files:
        print("\nTraining files (first 3):")
        for f in train_files[:3]:
            print(f"  {f.name}")
    else:
        print("\nNo training files found")

    if val_files:
        print("\nValidation files (first 3):")
        for f in val_files[:3]:
            print(f"  {f.name}")
    else:
        print("\nNo validation files found")

    if test_files:
        print("\nTest files (first 3):")
        for f in test_files[:3]:
            print(f"  {f.name}")
    else:
        print("\nNo test files found")

    # Demonstrate random file selection if we have training files
    if train_files:
        print("\nRandom training file example:")
        random_file = data_manager.get_random_training_file()
        print(f"  {random_file.name}")


if __name__ == "__main__":
    main()
