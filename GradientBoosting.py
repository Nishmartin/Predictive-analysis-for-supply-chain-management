import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv"
df = pd.read_csv(file_path)

# Select features and target variable
features = ['Sales', 'Order Item Quantity', 'Benefit per order']
target_variable = 'Order Profit Per Order'

X = df[features]
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting regressor
model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train Accuracy:", train_accuracy, "%")
print("Test Accuracy:", test_accuracy, "%")
