from Preprocessing import preprocess_data
from Graph import handle_errors  # Import the error handling function


def main():
    # Load and preprocess the data
    file_path = r'C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset.csv'

    try:
        df = preprocess_data(file_path)
        updated_file_path = r'C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv'
        df.to_csv(updated_file_path, index=False)
        print("Updated cleaned dataset saved successfully:", updated_file_path)
    except ValueError as e:
        handle_errors(e)
        exit(1)  # Terminate the script if an error occurs during preprocessing
    except Exception as e:
        handle_errors(e)
        exit(1)  # Terminate the script if an error occurs while saving the dataset


if __name__ == "__main__":
    main()
