import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv"
df = pd.read_csv(file_path)

# Define features and target variable
features = ['Sales', 'Order Item Quantity', 'Benefit per order']
target_variable = 'Order Profit Per Order'

# Prepare the data
X = df[features]
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the decision tree regressor
model = DecisionTreeRegressor(random_state=42)

# Initialize grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Fit the best model
best_model.fit(X_train, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

# Calculate and print accuracy percentage
train_accuracy = best_model.score(X_train, y_train) * 100
test_accuracy = best_model.score(X_test, y_test) * 100

print("Best Parameters:", best_params)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train Accuracy:", train_accuracy, "%")
print("Test Accuracy:", test_accuracy, "%")
