import sys
import pandas as pd
import streamlit as st
from pathlib import Path

# Add the parent directory to the system path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import the model from the .py file
from DecisionTree import best_model  # Assuming your best model is defined as 'best_model' in DecisionTree.py

# Load the dataset
file_path = r"C:/Users/Merline/Desktop/FYP/FinalYearProject2/cleaned_dataset_updated.csv"
df = pd.read_csv(file_path)

# Define features and target variable
features = ['Sales', 'Order Item Quantity', 'Benefit per order']
target_variable = 'Order Profit Per Order'

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
    prediction = best_model.predict(user_input)
    # Display prediction
    st.write(f"Predicted Order Profit Per Order: {prediction[0]}")
