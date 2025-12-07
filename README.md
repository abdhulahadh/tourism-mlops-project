
# Visit with Us - Wellness Tourism Prediction (MLOps)

## Business Context
**"Visit with Us"** is introducing a new **Wellness Tourism Package**. To optimize marketing costs and increase conversion rates, this project implements an end-to-end **MLOps pipeline** to predict which customers are most likely to purchase the package based on their profile and interaction history.

## Project Objective
Automate the machine learning lifecycle—from data ingestion to deployment—using **GitHub Actions**, **Hugging Face**, and **Docker**.

## Project Links
* **Live Web Application:** [Click Here to Access App](https://huggingface.co/spaces/abdhulahadh/tourism-prediction-app)
* **Dataset Source:** [Hugging Face Dataset](https://huggingface.co/datasets/abdhulahadh/tourism-dataset)
* **Model Registry:** [Hugging Face Model Hub](https://huggingface.co/abdhulahadh/tourism-model)

## Tech Stack
* **Experimentation:** MLflow, Scikit-learn, XGBoost
* **Data Versioning:** Hugging Face Datasets
* **Containerization:** Docker
* **CI/CD:** GitHub Actions
* **Frontend:** Streamlit

## Repository Structure
* `src/`: Python scripts for automation (`data_prep.py`, `train.py`)
* `data/`: Local storage for training data
* `model_building/`: Model artifacts
* `deployment/`: Dockerfile and Streamlit app code
* `.github/workflows/`: CI/CD pipeline configuration

## Model Performance
The project compares **Random Forest** and **XGBoost**.
* **Champion Model:** XGBoost
* **Performance Metric:** F1-Score (Chosen to handle class imbalance)
* **Result:** ~0.76 F1 Score / ~92% Accuracy

