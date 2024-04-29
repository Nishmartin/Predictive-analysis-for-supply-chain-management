import sys
import pandas as pd
import streamlit as st
from pathlib import Path
import os

# Add the parent directory to the system path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import the model from the .py files
from DecisionTree import best_model  # Assuming your best model is defined as 'best_model' in DecisionTree.py

# Load the dataset
file_path = os.path.join('cleaned_dataset_updated.csv')
df = pd.read_csv(file_path)

# Set title
st.title("Supply Chain Predictive Analytics")
# Add a sidebar
st.sidebar.title("Input Parameters")

# Select Target Variable
target_variable = st.sidebar.selectbox("Select Target Variable", df.columns)

# Select Features
selected_features = st.sidebar.multiselect("Select Features", df.columns)

# Input Values for Features
feature_values = {}
for feature in selected_features:
    if df[feature].dtype == 'object':
        feature_values[feature] = st.sidebar.selectbox(f"Select {feature}", df[feature].unique())
    else:
        feature_values[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

# Create a button to make predictions
if st.sidebar.button("Predict"):
    # Create a DataFrame with user inputs
    user_input_df = pd.DataFrame(feature_values, index=[0])
    # Make prediction
    prediction = best_model.predict(user_input_df)
    # Display prediction
    st.write(f"Predicted {target_variable}: {prediction[0]}")
