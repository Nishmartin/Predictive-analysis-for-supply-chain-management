import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_rmse = mean_squared_error(y_train, model.predict(X_train_scaled), squared=False)
test_rmse = mean_squared_error(y_test, model.predict(X_test_scaled), squared=False)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
