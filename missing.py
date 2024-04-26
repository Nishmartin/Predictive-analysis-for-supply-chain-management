import pandas as pd
df = pd.read_csv(r"C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv")

# Check for missing values
missing_values = df.isna().sum()
print("Missing Values:")
print(missing_values)