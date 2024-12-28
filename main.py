import sys
import data_processing
import lnscc_train


def main():
    try:
        print("Starting data processing...\n")
        data_processing.main()
        print("\nData processing completed successfully.\n")

        print("Starting training...\n")
        lnscc_train.main()
        print("\nTraining completed successfully.\n")

        print("All tasks have been executed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
