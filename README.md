# ClaimGuard: Intelligent Healthcare Service Pattern Analysis
![Python](https://img.shields.io/badge/python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-enabled-brightgreen)
![PySpark](https://img.shields.io/badge/PySpark-enabled-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-success)
![Transformers](https://img.shields.io/badge/HuggingFace-Bio_ClinicalBERT-blueviolet)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Optimized-informational)

![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![ML Pipeline](https://img.shields.io/badge/Pipeline-ML%20End--to--End-blue)
![Embeddings](https://img.shields.io/badge/NLP-Embeddings-lightgrey)
![MLOps](https://img.shields.io/badge/MLOps-Practices-important)

## 🎯 Project Goal

The goal of this project was to **predict the Medicare Allowed Amount** using a hybrid modeling approach that combines structured claim-level data with unstructured service descriptions.

By integrating **clinical language models**, **Delta Lake architecture**, and **MLOps practices**, this solution supports:

- Intelligent service categorization  
- Billing behavior prediction  
- Fraud detection foundations  
- Data integrity and auditability at scale  

This system provides a foundation for smarter decision-making and operational efficiency in healthcare claims processing.

---
## 🚀 Project Overview

- **Objective**: Predict the average allowed Medicare amount for healthcare services using both structured claim-level data and unstructured HCPCS descriptions.
- **Data Size**: 3.5GB CMS dataset, sampled to 200K records for training.
- **Output**: Predictive model with an R² score of ~0.95, explainability via SHAP, versioning via MLflow, and semantic analysis using Bio_ClinicalBERT.
---
## 🔗 Data Access
The Medicare Physician & Other Practitioners by Provider and Service dataset provides information on use, payments, and submitted charges organized by National Provider Identifier (NPI), Healthcare Common Procedure Coding System (HCPCS) code, and place of service.
<br>
- **Dataset Title**: Medicare Physician & Other Practitioners - by Provider and Service  
- **Source**: [CMS.gov Open Data Portal](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service/data/2021)
- **Direct API Access**:  
  Easily integrate this dataset programmatically using the following API endpoint:  
  [`https://data.cms.gov/data-api/v1/dataset/31dc2c47-f297-4948-bfb4-075e1bec3a02/data`](https://data.cms.gov/data-api/v1/dataset/31dc2c47-f297-4948-bfb4-075e1bec3a02/data)
---
## 1. Data Preprocessing & Cleaning

We began with the **CMS Medicare Part B dataset**, containing over **9.7 million rows** and **28 columns**, including provider details, service information, and billing amounts.

### Preprocessing Steps
  - **Handled Null Values**  
  Used imputation strategies (e.g., filling with median or constants) to ensure no missing data affected model training.
  - **Geographic Simplification**  
  Extracted and simplified ZIP codes and state information to avoid high-cardinality location features.
  - **Aggregated Metrics**  
  Calculated provider-level aggregates like total services, beneficiaries, and statistical measures of cost (mean, std, max).
- **Categorical Encoding**  
  Applied Label Encoding to fields like `Provider_Type`, `Place_Of_Service`, and `Medicare_Participation`.

- **Financial Feature Engineering**  
  Introduced new features like:
  - `Charge_to_Allowed_Ratio`
  - `Payment_to_Allowed_Ratio`
  - `Num_Unique_Procedures`  
  These features help capture billing behavior patterns effectively.

- **Removed Duplicates**  
  Ensured data integrity before embedding and training.

- **Saved Cleaned Dataset**  
  Final cleaned file saved as `cleaned_claimguard_data.parquet` for embedding and modeling phases.

### Why This Was the Best Approach

- **Parquet + PySpark** enabled scalable, memory-efficient processing of 9M+ records.
- **Aggregating at the provider level** helped generalize billing behavior over time.
- **Label encoding** ensured compatibility with XGBoost without inflating feature dimensions.
- **Financial features** added meaningful variance and predictive power to the model.
- The cleaned dataset served as a robust foundation for embeddings, model training, and interpretability work ahead.
---
## 🗺️ Geographic Insights: Number of Providers by State
![image](https://github.com/user-attachments/assets/edb9346a-ba88-4b66-9ca9-ba57225dc89a)
To better understand provider distribution across the U.S., we visualized the number of providers per state using a choropleth map. This helps identify regions with high or low healthcare access, informing policy and outreach decisions.

### 🧠 Key Takeaways:

- **High-density states** like **California, Texas, Florida, and New York** have the most providers — aligned with their large populations and healthcare demand.
- **Mid-sized states** such as **Illinois** and **Georgia** serve as regional healthcare hubs.
- **Rural states** like **Montana**, **Wyoming**, and **Vermont** show significantly fewer providers, indicating potential access challenges.
- This map supports the need for **targeted healthcare policies** and **strategic resource allocation**.

---
## 🧊 Delta Lake & Storage Layer

After preprocessing and cleaning the CMS healthcare dataset, we transitioned the data into a more robust and production-ready format using **Delta Lake**.

### ✅ What We Did

- **Configured PySpark and Delta Lake** to operate in a scalable, cloud-compatible environment.
- **Converted cleaned Parquet files to Delta format**, enabling transactional consistency.
- Ran **aggregations and analytical queries** to understand provider-level billing behaviors.
- Applied **row-level operations** like `UPDATE` and `DELETE` for data correction and evolution.
- Enabled **schema evolution** to accommodate new fields without breaking the pipeline.
- Simulated **time travel queries** for versioned snapshots and rollback testing.
- Supported **batch appends** of new claim records while maintaining schema integrity.

### 💡 Why This Approach Was Critical

Traditional formats like CSV or plain Parquet do not support ACID transactions, historical versioning, or schema enforcement — all essential for production-grade pipelines.

Delta Lake allows us to:
- **Update and clean existing data** post-ingestion.
- **Track changes over time** using data versioning.
- **Run reliable queries** without data corruption from partial writes.

### 📈 How This Helped the Project

Delta Lake serves as the **single source of truth** for all downstream ML components:

- **Faster I/O** performance for large datasets (9.7M rows).
- **Reliable auditing and rollback support** via time travel.
- **Enables clean feature engineering and model retraining** without redundant full reloads.

### 🔗 Project Integration

The resulting **Delta table** is now used as the foundational data store for the next stages:
- Embedding generation using Transformer-based models (e.g., BioBERT)
- Feature selection and billing behavior prediction via XGBoost
- Model monitoring and retraining pipelines powered by MLflow

---
## 🤖 Embedding Generation, Model Training & MLOps (MLflow)

The core objective of this stage was to build a reliable ML pipeline to predict Medicare billing behavior using structured features and semantic embeddings from medical procedure descriptions.

---

### ✅ What We Did

- **Sampled 200k rows** from the cleaned Delta-backed dataset to ensure memory-efficient training.
- Used **Bio_ClinicalBERT** from Hugging Face to generate 768-dim embeddings from `HCPCS_Description`.
- Merged **structured features** (e.g., geographic, financial, categorical) with **LLM-based embeddings**.
- Performed a **train-test split** and trained an **XGBoost Regressor** on the target column: `Average_Medicare_Allowed_Amount`.
- Evaluated model with an **R² score of 0.95**, validating strong predictive performance.
- Logged predictions vs. actual values for quick validation and trust-building.

---

### 🧠 Explainability with SHAP

![image](https://github.com/user-attachments/assets/af98b91d-8fde-45ff-949d-899950cdce8d)

- Computed **top 20 feature importances** using XGBoost's built-in gain metric.
- Used **SHAP explainer plots** to understand how each feature (including embeddings) influenced predictions.
- Delivered both **global** and **local interpretability**, enhancing trust in model predictions.

---

### ⚙️ MLOps with MLflow

- Set up **MLflow experiment tracking**, logging:
  - Hyperparameters (e.g., learning rate, max depth)
  - Model artifacts
  - Evaluation metrics (R²)
- Registered the trained model in the **MLflow Model Registry** for future use.
- Implemented **real-time prediction pipelines**:
  - Dynamically built feature dictionaries
  - Generated fresh BioBERT embeddings
  - Made predictions using the registered model

---

### 📦 Logged Artifacts & Outputs

- `xgb_claimguard_model.pkl` – Trained XGBoost model
- `single_prediction.csv` – Output from one prediction sample
- `prediction_output.csv` – Batch predictions for validation
- MLflow logs include **metrics, parameters, SHAP plots, and predictions**

---

### 🔁 Why This Pipeline Matters

- Leverages **domain-specific transformer embeddings (BioBERT)** for better context from `HCPCS_Description`.
- Combines **deep learning with tabular ML** using XGBoost — best of both worlds.
- MLOps integration ensures the system is **trackable, reproducible, and deployable**.
- Ideal for **real-world healthcare monitoring systems** where transparency and prediction consistency are essential.

---

### 🚀 What’s Next

- Extend to classification use cases (e.g., High-cost vs. Preventive care)
- Automate embedding refresh cycles for new codes
- Integrate drift detection and retraining triggers in MLflow

---
## 🎓  What I Learned & Why This Project Matters
Through **ClaimGuard**, I gained hands-on experience in end-to-end data science and MLOps workflows, including:

- Processing and analyzing large-scale healthcare datasets (9.7M+ rows)
- Applying NLP techniques **(Bio_ClinicalBERT) to derive semantic embeddings** from HCPCS descriptions
- Training interpretable and high-performance models using **XGBoost**
- Implementing Delta Lake for scalable, version-controlled storage with update/delete support
- Applying SHAP for model explainability and feature attribution
- Logging, tracking, and managing model versions using **MLflow and Model Registry**
- Creating real-time prediction pipelines using structured + unstructured inputs

