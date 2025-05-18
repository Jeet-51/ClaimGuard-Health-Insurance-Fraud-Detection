# ClaimGuard: Intelligent Healthcare Service Pattern Analysis
ClaimGuard is an end-to-end machine learning pipeline designed to analyze Medicare Part B claims data, generate semantic embeddings from medical service descriptions, predict billing behavior, and monitor model performance using MLOps best practices.

---

## 🚀 Project Overview

- **Objective**: Predict the average allowed Medicare amount for healthcare services using both structured claim-level data and unstructured HCPCS descriptions.
- **Data Size**: 3.5GB CMS dataset, sampled to 200K records for training.
- **Output**: Predictive model with an R² score of ~0.95, explainability via SHAP, versioning via MLflow, and semantic analysis using Bio_ClinicalBERT.

---

## 🧱 Key Components

### 🧾 Data
- Format: `.parquet`
- Columns: Structured provider info, billing fields, HCPCS_Description
- Preprocessing: Encoding, ratio engineering, embeddings

### 🤖 Embedding Generation
- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Use Case**: Transforms HCPCS_Description into 768-dim semantic vectors
- **Tools**: HuggingFace Transformers, PyTorch

### 🔍 Model Training
- **Model**: XGBoost Regressor
- **Target**: `Average_Medicare_Allowed_Amount`
- **Metrics**: R² = 0.955+, MAE, MAPE, RMSE
- **Tools**: `xgboost`, `scikit-learn`

### 📈 Explainability
- **Method**: SHAP (SHapley Additive Explanations)
- **Visuals**: Summary plot, waterfall plot
- **Insights**: Mix of structured & embedding dimensions drive predictions

### ⚙️ MLOps & Tracking
- **Platform**: MLflow
- **Tracking**: Parameters, metrics, model versions, prediction logs
- **Artifacts**: `.csv` inputs, model registry, single prediction logging

---

## 🛠 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
