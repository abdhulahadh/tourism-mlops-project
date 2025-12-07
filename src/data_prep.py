
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi
import os

# CONFIG
HF_USERNAME = "abdhulahadh" 
DATASET_NAME = "tourism-dataset"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

print(" [CI/CD] Starting Data Preparation")

# Load
print(f" Downloading data from {REPO_ID}")
file_path = hf_hub_download(repo_id=REPO_ID, filename="tourism.csv", repo_type="dataset")
df = pd.read_csv(file_path)

# Clean
if 'CustomerID' in df.columns: df.drop(columns=['CustomerID'], inplace=True)
if 'Unnamed: 0' in df.columns: df.drop(columns=['Unnamed: 0'], inplace=True)
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['number']).columns
for col in cat_cols: df[col] = df[col].fillna(df[col].mode()[0])
for col in num_cols: df[col] = df[col].fillna(df[col].median())

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'])

# Save
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Upload
api = HfApi()
api.upload_file(path_or_fileobj="data/train.csv", path_in_repo="train.csv", repo_id=REPO_ID, repo_type="dataset")
api.upload_file(path_or_fileobj="data/test.csv", path_in_repo="test.csv", repo_id=REPO_ID, repo_type="dataset")
print("[CI/CD] Data Preparation Complete & Uploaded")
