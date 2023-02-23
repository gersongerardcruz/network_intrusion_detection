import argparse
import streamlit as st
import requests
import pandas as pd
import json
import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

# Define function to make API request and return predictions
def get_predictions(file_bytes):

    # Define the FastAPI endpoint URL
    api_url = args.endpoint

    # Make API request
    response = requests.post(api_url, files={'file': file_bytes})

    # Check if request was successful
    if response.status_code == 200:
        # Parse predictions from response
        predictions = response.json()

        # Return predictions
        return predictions

    # If request failed, raise an exception
    else:
        raise Exception('API request failed with status code {}'.format(response.status_code))

# Define the Streamlit app
def app():
    
    # Set app title
    st.title('Deployment for the End-to-End Network Intrusion Detection Model')

    # Define a placeholder for the uploaded file data
    file_data = None

    # Define a placeholder for the predictions
    predictions = None

    # Use st.file_uploader to get a file from the user
    file = st.file_uploader('Choose file to upload', type=['csv', 'xlsx'])

    # If a file was uploaded
    if file is not None:
        # Use pandas to read the file into a dataframe
        file_data = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

        file_data = remove_id_column(file_data)

        # Convert the file data to bytes
        file_bytes = file.getvalue()

        # Write the dataframe head to the app for the user to check
        st.write(file_data.head())

    # If the user has uploaded a file and wants to make predictions
    if file_data is not None and st.button('Predict'):

        st.write("Prediction in progress...")

        # Use st.progress to display a progress bar while predictions are being made
        progress_bar = st.progress(0)

        # Make API request to get predictions
        predictions = get_predictions(file_bytes)

        # Update progress bar to 100%
        progress_bar.progress(100)

    # If predictions have been made, display a download button for the predictions
    if predictions is not None:
        # Display prediction results
        st.write("Prediction successful! Here are the first five results: ")
        st.json({k: predictions[k] for k in list(predictions)[:5]})
        
        # Use st.download_button to allow the user to download the predictions as a CSV file
        st.download_button(
            label='Download predictions',
            data=json.dumps(predictions),
            file_name='predictions.json',
            mime='application/json'
        )

# Run the Streamlit app
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", help="endpoint for prediction", default='http://localhost:8000/predict')
    args = parser.parse_args()

    app()








