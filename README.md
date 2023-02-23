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

The technical team can generate value from an AutoML-based solution by: 

1. Saving time and resources: AutoML automates the process of model selection, hyperparameter tuning, and feature engineering, which saves time and resources for the organization.
2. Increasing accuracy: AutoML uses advanced algorithms to optimize the performance of machine learning models, which can lead to higher accuracy and more reliable predictions.
3. Enabling experimentation: AutoML allows organizations to quickly experiment with different models, which can lead to new insights and ideas that may not have been possible otherwise.
4. Improving scalability: AutoML can scale up or down based on the data size and complexity, which enables organizations to handle large datasets and complex models efficiently.

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
    │   ├── backend
    │   ├── frontend