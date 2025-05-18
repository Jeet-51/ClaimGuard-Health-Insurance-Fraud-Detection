# ClaimGuard: Intelligent Healthcare Service Pattern Analysis
ClaimGuard is an end-to-end machine learning pipeline designed to analyze Medicare Part B claims data, generate semantic embeddings from medical service descriptions, predict billing behavior, and monitor model performance using MLOps best practices.

---

## 🚀 Project Overview

- **Objective**: Predict the average allowed Medicare amount for healthcare services using both structured claim-level data and unstructured HCPCS descriptions.
- **Data Size**: 3.5GB CMS dataset, sampled to 200K records for training.
- **Output**: Predictive model with an R² score of ~0.95, explainability via SHAP, versioning via MLflow, and semantic analysis using Bio_ClinicalBERT.

---
## 1. Data Preprocessing & Cleaning

We began with the **CMS Medicare Part B dataset**, containing over **9.7 million rows** and **28 columns**, including provider details, service information, and billing amounts.

### What We Did

- **Converted to Parquet Format**  
  Efficient, compressed storage for fast reads/writes and compatibility with PySpark.

- **Renamed Columns**  
  Made column names consistent and readable for downstream modeling.

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

