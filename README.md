# End-to-End Network Intrusion Detection using AutoML, Mlflow, Streamlit, and FastAPI

## Project Information

Network intrusion detection is a critical aspect of cybersecurity that helps organizations safeguard their networks and sensitive data from cyber attacks. As more businesses and individuals rely on technology for daily operations and communication, the risk of cyber attacks becomes increasingly prevalent. Malicious actors use various techniques to exploit vulnerabilities in networks and gain unauthorized access to confidential information, which can result in financial loss, legal ramifications, and reputational damage. 

Network intrusion detection models serve as a powerful defense mechanism against these threats by monitoring network traffic and identifying potential security breaches. By providing early warning signals of potential attacks, these models allow security teams to take immediate action and prevent or mitigate the impact of a cyber attack. 

Given the importance of protecting sensitive data in today's digital landscape, network intrusion detection models are essential for maintaining the integrity and security of networks and systems. Thus, this is an **end-to-end deployment of a network intrusion classification model** trained on [Kaggle's Network Intrusion Detection Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection) using four key components: **AutoML, Mlflow, Streamlit, and FastAPI**. 

## Business Value

Businesses using network intrusion detection models can derive value by having: 
1. Improved security: By accurately identifying and flagging anomalous network activity, organizations can better protect themselves against potential cyber threats, such as malware, viruses, and hacking attempts.
2. Reduced downtime: Early detection of anomalous network activity can help prevent system crashes and other downtime events, improving overall system uptime and reducing potential losses in productivity and revenue.

3. Increased efficiency: By automating the process of network intrusion detection, organizations can reduce the need for manual review and analysis of network logs, freeing up time and resources for other business-critical tasks.

## Technical Value

The technical team can generate value from utilizing AutoML and Mlflow in the development of the network instruction model because of the following. 

### AutoML:

* Saves time and effort by automating the selection of the best algorithm and hyperparameters for the network intrusion detection model.
* Reduces the risk of human error in the model selection process by automating it.
* Increases the accuracy of the network intrusion detection model by optimizing the algorithm and hyperparameters for the dataset.

### Mlflow:

* Makes it easier to reproduce experiments and results by tracking the inputs, parameters, and metrics used in the network intrusion detection model.
* Enables effective collaboration and knowledge sharing among data scientists and engineers by providing a centralized repository for model experimentation and results.
* Provides version control for models and datasets, allowing for easy comparison and selection of the best model for deployment.
* Facilitates the deployment of the network intrusion detection model by generating deployable artifacts and providing an API for serving the model.

## Project Structure

    ├── README.md          <- The top-level documentation for this project.
    ├── data
    │   ├── processed      <- The final data sets for customer segmentation.
    │   ├── interim        <- Folder for holding data with intermediate transformations
    │   └── raw            <- The original, immutable datasets.
    ├── images             <- The media used in the README documentation
    ├── notebooks          <- Jupyter notebooks containing the explorations performed in this project
    ├── references         <- Folder containing relevant files for keeping track of trained models and features
    ├── requirements.txt   <- The requirements file for reproducing the project
    ├── src                <- Folder containing all source code in the project
    │   ├── backend        <- Folder for all files for setting up the backend 
    │   ├── frontend       <- Folder for all files for setting up the frontend

## Key Project Files

- `data/`: Folder containing all datasets (training, interim, processed). Default: interim and processed folders empty, to be populated by running `preprocessing.py`.
    - `train/`: Folder containing raw data taken from Kaggle
- `notebooks`: Folder containing jupyter notebooks for the project
    - `01_eda.ipynb`: notebook performing eda and interim cleaning of raw data
- `references/`: Folder containing files for tracking models and features
    - `train_processed_column_types.json`: json file containing all columns and their respective types
    - `leaderboard.csv`: csv file for the ranking of models and their performance
    - `cv_summary.csv`: csv file containing information regarding cv-fold metrics of the best model
- `src`: Folder containing source code for both backend and frontend
    - `utils.py`: Python file containing all helper functions for preprocessing, training, and deployment
    - `backend/`: Folder containing all files for setting up the backend (preprocessing, AutoML training, FastAPI)
        - `preprocess.py`: Python file for performing preprocessing steps such as categorical encoding and numerical scaling 
        - `train.py`: Python file for performing AutoML training tracked using Mlflow
        - `main.py`: Python file containing FastAPI post request to make predictions and selecting the best trained model for it
    - `frontend/`: Folder containing all files for frontend components (Streamlit UI)
        - `app.py`: Python file for spinning up the Streamlit application for uploading test data, making predictions, and downloading predictions


    
 
