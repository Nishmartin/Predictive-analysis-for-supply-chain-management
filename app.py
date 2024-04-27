import pandas as pd
import streamlit as st
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

# Initialize the decision tree regressor
model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=1, random_state=42)  # Use the best parameters obtained from grid search

# Fit the model
model.fit(X, y)

# Set title
st.title("Supply Chain Predictive Analytics")

# Add a sidebar
st.sidebar.title("Input Parameters")

# Add input fields for features
sales = st.sidebar.number_input("Sales", value=0.0)
quantity = st.sidebar.number_input("Order Item Quantity", value=0)
benefit = st.sidebar.number_input("Benefit per order", value=0.0)

# Create a button to make predictions
if st.sidebar.button("Predict"):
    # Create a DataFrame with user inputs
    user_input = pd.DataFrame({"Sales": [sales], "Order Item Quantity": [quantity], "Benefit per order": [benefit]})
    # Make prediction
    prediction = model.predict(user_input)
    # Display prediction
    st.write(f"Predicted Order Profit Per Order: {prediction[0]}")