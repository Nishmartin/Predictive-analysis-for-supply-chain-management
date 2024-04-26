import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(['Customer Email', 'Customer Password', 'Product Image', 'Product Name', 'Order Id'], axis=1)

    # Convert object type columns to category type
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = df[col].astype('category')

    # Label encode categorical variables
    label_encoders = {}
    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


def train_model(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Preprocess the data
    preprocessed_df, _ = preprocess_data(df)

    # Define features and target variable
    X = preprocessed_df.drop(['Order Profit Per Order'], axis=1)
    y = preprocessed_df['Order Profit Per Order']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model
    model = XGBRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

    # Calculate and print accuracy percentage
    train_accuracy = model.score(X_train, y_train) * 100
    test_accuracy = model.score(X_test, y_test) * 100

    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Train Accuracy:", train_accuracy, "%")
    print("Test Accuracy:", test_accuracy, "%")


# Example usage:
file_path = r"C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv"
train_model(file_path)
