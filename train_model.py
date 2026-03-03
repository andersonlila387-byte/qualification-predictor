"""
Script to train and save the model
"""
import os
import sys

# Change to model directory
os.chdir('c:/Users/DELL/Documents/model')

# Print debug info
print(f"Current directory: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')}")

from src.model import initialize_model

# Initialize and train the model
print("Training model...")
predictor = initialize_model()
print("Model trained and saved successfully!")
print(f"Model path: {predictor.model_path}")
print(f"Scaler path: {predictor.scaler_path}")

# Verify files exist
print(f"Model file exists: {os.path.exists(predictor.model_path)}")
print(f"Scaler file exists: {os.path.exists(predictor.scaler_path)}")
