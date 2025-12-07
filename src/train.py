
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import hf_hub_download, HfApi
import os

# CONFIG
HF_USERNAME = "abdhulahadh"
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO = f"{HF_USERNAME}/tourism-model"

print("[CI/CD] Starting Model Training & Tuning")

# Load Data
train_path = hf_hub_download(repo_id=DATASET_REPO, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=DATASET_REPO, filename="test.csv", repo_type="dataset")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken']
X_test = test_df.drop(columns=['ProdTaken'])
y_test = test_df['ProdTaken']

# Define Pipeline
categorical_features = X_train.select_dtypes(include=['object']).columns
numeric_features = X_train.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Models to Tune (Matches Notebook)
models_to_tune = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [10, 20]}
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "params": {"classifier__learning_rate": [0.1], "classifier__max_depth": [6]}
    }
}

best_overall_model = None
best_overall_score = -1
best_model_name = ""

# Train & Track
mlflow.set_experiment("Tourism_CI_CD_Run")

for name, config in models_to_tune.items():
    with mlflow.start_run(run_name=name):
        print(f"Tuning {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', config["model"])])
        
        # Grid Search
        grid_search = GridSearchCV(pipeline, config["params"], cv=2, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_run_model = grid_search.best_estimator_
        y_pred = best_run_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        print(f" {name} Best F1: {f1:.4f}")
        
        if f1 > best_overall_score:
            best_overall_score = f1
            best_overall_model = best_run_model
            best_model_name = name

print(f" Chosen Model: {best_model_name} (F1: {best_overall_score:.4f})")

# Save & Register
os.makedirs("model_building", exist_ok=True)
joblib.dump(best_overall_model, "model_building/model.joblib")

api = HfApi()
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="model_building/model.joblib", path_in_repo="model.joblib", repo_id=MODEL_REPO, repo_type="model")
print(" [CI/CD] Best Model Registered to HF Hub")
