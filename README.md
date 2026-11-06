# E-Commerce Product Return Prediction System using CI/CD

An end-to-end Machine Learning Operations (MLOps) project that predicts whether an e-commerce product order will be returned or retained. The project integrates automated data preprocessing, feature engineering, dual-model training (Logistic Regression vs. Random Forest), model selection based on F1-score, FastAPI deployment, Pytest test automation, containerization with Docker, and automated CI/CD using GitHub Actions publishing to GitHub Container Registry (GHCR).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Directory Structure](#project-directory-structure)
3. [Dataset & Data Preprocessing](#dataset--data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Pipeline & Evaluation](#machine-learning-pipeline--evaluation)
6. [FastAPI Microservice](#fastapi-microservice)
7. [Automated Testing Suite (Pytest)](#automated-testing-suite-pytest)
8. [Docker Containerization](#docker-containerization)
9. [GitHub Actions CI/CD Pipeline](#github-actions-cicd-pipeline)
10. [Step-by-Step GitHub & GHCR Deployment Guide](#step-by-step-github--ghcr-deployment-guide)
11. [Examiner Practical Examination Checklist & Screenshots Guide](#examiner-practical-examination-checklist--screenshots-guide)

---

## Project Overview

In e-commerce operations, product returns cause significant logistics costs and inventory depreciation. This system provides real-time return probability scoring and categorical prediction (`Returned` vs `Not Returned`) for incoming orders.

### Key Highlights
- **End-to-End MLOps**: Seamless path from raw examiner dataset to trained model artifacts, REST API, automated testing, container packaging, and CI/CD.
- **Robust Feature Engineering**: Custom delivery delay metric, historical customer return frequency, and range validation for discounts and pricing.
- **Automated Model Selection**: Evaluates Logistic Regression and Random Forest on an identical stratified test split, selecting the champion model based on **F1-score**.
- **Containerized REST API**: Production-ready FastAPI application with health check (`/health`), batch/single prediction (`/predict`), interactive OpenAPI documentation (`/docs`), and CORS middleware.
- **Automated CI/CD**: GitHub Actions workflow that executes syntax validation, runs 11 automated Pytest unit/integration tests, builds a Docker image, and publishes it securely to GitHub Container Registry (`ghcr.io`).

---

## Project Directory Structure

```text
mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD workflow
├── app/
│   ├── __init__.py            # Python package indicator
│   ├── main.py                # FastAPI endpoints (GET /health, POST /predict)
│   ├── schemas.py             # Pydantic request & response data contracts
│   └── predictor.py           # Model loading & inference pipeline
├── data/
│   ├── __init__.py
│   └── Section7_Q4_Ecommerce_Product_Return_CICD.csv  # Examiner dataset
├── model/
│   ├── __init__.py
│   ├── preprocessor.py        # Data cleaning & feature engineering transformers
│   ├── train.py               # Model training, evaluation & artifact persistence
│   ├── best_model.joblib      # Serialized champion model pipeline
│   └── metrics.json           # Evaluation metrics for all candidate models
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # FastAPI endpoint tests (health, predict, errors)
│   ├── test_model.py          # Model artifact integrity & inference tests
│   └── test_preprocessing.py  # Unit tests for preprocessing & feature math
├── .dockerignore              # Files excluded from Docker context
├── .gitignore                 # Files excluded from Git version control
├── Dockerfile                 # Multi-stage production container definition
├── pytest.ini                 # Pytest runner configuration
├── requirements.txt           # Python application dependencies
└── README.md                  # Comprehensive documentation and exam report
```

---

## Dataset & Data Preprocessing

The project uses the provided dataset: `Section7_Q4_Ecommerce_Product_Return_CICD.csv` (1,200 samples, 14 columns).

### Target Variable
- **Target Column**: `Return_Status`
  - Values: `Not Returned` (929 samples, 77.4%), `Returned` (271 samples, 22.6%).
  - Encoded into binary: `Returned` = 1, `Not Returned` = 0.

### Preprocessing & Validation Logic
1. **Product_Price_INR Validation**:
   - Negative prices (e.g., `-200.0` found in raw data) are flagged as invalid and converted to `NaN`.
   - Missing prices are imputed using **Median** imputation.
2. **Discount_Percent Validation**:
   - Out-of-bounds discount percentages (e.g., `125.0%` found in raw data, values `< 0` or `> 100`) are flagged as invalid and converted to `NaN`.
   - Missing discounts are imputed using **Median** imputation.
3. **Delivery Days & Ratings**:
   - Missing `Actual_Delivery_Days` and `Customer_Rating_PreOrder` are imputed using median values.
4. **Categorical Features**:
   - `Product_Category`, `Payment_Method`, `Damaged_Package_Flag`, `Package_Size`, `Promotion_Active`.
   - Missing categories (e.g. `Payment_Method`) are imputed using **Most Frequent** imputation.
   - Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
5. **Numerical Features**:
   - Scaled using `StandardScaler` to ensure zero mean and unit variance.

---

## Feature Engineering

Two key business-critical engineered features are created during preprocessing:

1. **Delivery_Delay_Days**:
   $$\text{Delivery\_Delay\_Days} = \text{Actual\_Delivery\_Days} - \text{Expected\_Delivery\_Days}$$
   - Positive value indicates delivery delay (associated with customer dissatisfaction and higher return likelihood).
   - Negative value indicates early delivery.

2. **Customer_Return_Frequency**:
   $$\text{Customer\_Return\_Frequency} = \frac{\text{Previous\_Returns}}{\max(\text{Previous\_Orders}, 1)}$$
   - Historical return ratio for the customer.
   - Guarded against division by zero for customers with zero previous orders.

3. **Discount_Percent Validation**:
   - Validated strictly within $[0, 100]$.

---

## Machine Learning Pipeline & Evaluation

The training script `model/train.py` trains two models on an 80/20 stratified split (`random_state=42`):
1. **Logistic Regression** (Balanced class weighting)
2. **Random Forest Classifier** (Balanced class weighting, 100 estimators)

### Evaluation Metrics (Verified Actual Results)

| Metric | Logistic Regression (Selected Best) | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | **0.6083** (60.83%) | **0.7125** (71.25%) |
| **Precision** | **0.2727** (27.27%) | **0.3171** (31.71%) |
| **Recall** | **0.4444** (44.44%) | **0.2407** (24.07%) |
| **F1-Score** *(Selection Metric)* | **0.3380** | **0.2737** |
| **ROC-AUC** | **0.5748** | **0.5275** |

### Champion Model Selection
- **Best Model**: `LogisticRegression`
- **Reason**: The objective is to identify product returns in an imbalanced dataset. Logistic Regression achieved an **F1-Score of 0.3380** (nearly 24% higher than Random Forest's 0.2737) and a significantly higher **Recall of 0.4444** (vs 0.2407), catching substantially more returns.
- **Persistence**: Saved to `model/best_model.joblib` and metrics written to `model/metrics.json`.

---

## FastAPI Microservice

FastAPI serves the champion pipeline with low latency and automatic Swagger documentation.

### Endpoints

#### 1. `GET /health`
Verifies API operational status and model loading.

**Response (HTTP 200)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "best_model": "LogisticRegression",
  "version": "1.0.0"
}
```

#### 2. `POST /predict`
Accepts order details, calculates feature engineering in real-time, and predicts return likelihood.

**Request Payload**:
```json
{
  "Product_Category": "Electronics",
  "Product_Price_INR": 22055.97,
  "Discount_Percent": 21.6,
  "Expected_Delivery_Days": 9,
  "Actual_Delivery_Days": 12.0,
  "Customer_Rating_PreOrder": 4.4,
  "Previous_Returns": 3,
  "Previous_Orders": 10,
  "Payment_Method": "Card",
  "Damaged_Package_Flag": "Yes",
  "Package_Size": "Large",
  "Promotion_Active": "Yes"
}
```

**Response (HTTP 200)**:
```json
{
  "prediction": "Returned",
  "return_probability": 0.6514,
  "return_status_code": 1,
  "model_used": "LogisticRegression",
  "engineered_features": {
    "Delivery_Delay_Days": 3.0,
    "Customer_Return_Frequency": 0.3
  }
}
```

---

## Automated Testing Suite (Pytest)

The project includes 11 comprehensive automated tests in `tests/`:

1. `tests/test_preprocessing.py`:
   - `test_validate_discount_percent`: Validates $[0, 100]$ bounds and invalid input handling.
   - `test_validate_product_price`: Validates $> 0$ pricing and invalid negative handling.
   - `test_feature_engineering_calculations`: Tests delay calculation and zero-division safeguard.
2. `tests/test_model.py`:
   - `test_model_artifact_exists`: Verifies `best_model.joblib` exists and exposes `predict()`.
   - `test_metrics_json_integrity`: Validates all 5 metrics for both candidate models.
   - `test_single_prediction_output`: Verifies prediction format and probability bounds $[0.0, 1.0]$.
3. `tests/test_api.py`:
   - `test_root_endpoint`: Tests `GET /`.
   - `test_health_endpoint`: Tests `GET /health`.
   - `test_predict_endpoint_valid`: Tests `POST /predict` standard scenario.
   - `test_predict_endpoint_high_risk_order`: Tests `POST /predict` high return risk.
   - `test_predict_endpoint_validation_error`: Tests `POST /predict` schema rejection (HTTP 422).

### Running Tests Locally
```bash
python -m pytest -v
```

---

## Docker Containerization

The application is containerized using `python:3.12-slim` with a built-in healthcheck probe.

### Build and Run Docker Locally
```bash
# 1. Build the Docker image
docker build -t ecommerce-return-prediction:latest .

# 2. Run the Docker container
docker run -d -p 8000:8000 --name return-prediction-api ecommerce-return-prediction:latest

# 3. Test the container endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Product_Category\":\"Beauty\",\"Product_Price_INR\":8772.58,\"Discount_Percent\":20.1,\"Expected_Delivery_Days\":10,\"Actual_Delivery_Days\":12.0,\"Customer_Rating_PreOrder\":4.1,\"Previous_Returns\":3,\"Previous_Orders\":18,\"Payment_Method\":\"Card\",\"Damaged_Package_Flag\":\"No\",\"Package_Size\":\"Large\",\"Promotion_Active\":\"No\"}"
```

---

## GitHub Actions CI/CD Pipeline

The CI/CD pipeline defined in `.github/workflows/ci-cd.yml` automates testing, code validation, Docker building, and package publishing.

### Pipeline Workflow Architecture
```
[Push to main / PR / Dispatch]
             │
             ▼
┌──────────────────────────────────────────────┐
│  Job 1: test-and-validate                    │
│  - Set up Python 3.12                        │
│  - Install dependencies from requirements.txt │
│  - Check Python compilation & Flake8 syntax   │
│  - Run 11 Pytest unit/integration tests      │
└──────────────────────┬───────────────────────┘
                       │ (Success)
                       ▼
┌──────────────────────────────────────────────┐
│  Job 2: build-and-publish (Main branch only) │
│  - Set up Docker Buildx                      │
│  - Log in to GitHub Container Registry (GHCR)│
│  - Build Docker image                        │
│  - Publish to ghcr.io/<owner>/<repo>:latest  │
└──────────────────────────────────────────────┘
```

### Security & Token Handling
- The pipeline utilizes `GITHUB_TOKEN` automatically provided by GitHub Actions with least-privilege scoping:
  ```yaml
  permissions:
    contents: read
    packages: write
  ```
- No personal access tokens (PAT) or plain passwords are hardcoded.

---

## Step-by-Step GitHub & GHCR Deployment Guide

Follow these steps to deploy this project to GitHub and verify the automated CI/CD pipeline.

### Step 1: Create a GitHub Repository
1. Log in to your GitHub account: [https://github.com](https://github.com).
2. Click the **+** icon in the upper right corner and select **New repository**.
3. Name your repository (e.g., `ecommerce-return-prediction-cicd`).
4. Set visibility to **Public** (recommended for easy GHCR access).
5. **Do NOT** initialize with a README, .gitignore, or license (we already have clean files prepared).
6. Click **Create repository**.

### Step 2: Configure Workflow Permissions in GitHub
1. In your GitHub repository, navigate to **Settings** > **Actions** > **General**.
2. Scroll down to **Workflow permissions**.
3. Select **Read and write permissions**.
4. Check **Allow GitHub Actions to create and approve pull requests** (if present).
5. Click **Save**.

### Step 3: Connect Local Repository and Push Code
Open your terminal in the project directory (`c:\Users\komal\OneDrive\ドキュメント\Desktop\mlops`) and run:

```bash
# 1. Add all project files
git add .

# 2. Commit the changes
git commit -m "feat: complete e-commerce return prediction system with CI/CD"

# 3. Add remote repository (replace <USERNAME> and <REPO_NAME> with your details)
git remote add origin https://github.com/<USERNAME>/<REPO_NAME>.git

# 4. Push to main branch
git push -u origin main
```

*(Note: We will request your explicit approval before pushing any code).*

### Step 4: Verify GitHub Actions Workflow
1. In your GitHub repository, click on the **Actions** tab.
2. You will see the workflow **CI/CD Pipeline - E-Commerce Return Prediction** running.
3. Click on the workflow run to view the two jobs:
   - `Code Validation & Automated Tests` (Compilation, Flake8, Pytest 11/11 tests passed).
   - `Build & Publish Docker Image to GHCR` (Docker build & push to `ghcr.io`).
4. Verify both jobs complete with green checkmarks.

### Step 5: Verify Image in GitHub Container Registry (GHCR)
1. On the main page of your GitHub repository, look at the right sidebar under **Packages**.
2. Click on the published package (e.g., `ecommerce-return-prediction-cicd`).
3. You will see the Docker image tags: `latest` and short commit SHA.

### Step 6: Pull GHCR Image and Run Container Locally
To verify the published image on any machine with Docker installed:

```bash
# 1. Pull the Docker image from GitHub Container Registry
docker pull ghcr.io/<USERNAME>/<REPO_NAME>:latest

# 2. Run the container
docker run -d -p 8000:8000 --name ghcr-return-prediction ghcr.io/<USERNAME>/<REPO_NAME>:latest

# 3. Verify health
curl http://localhost:8000/health

# 4. Access Interactive Documentation in Browser
# Open: http://localhost:8000/docs
```

---

## Examiner Practical Examination Checklist & Screenshots Guide

Use this checklist during your practical exam presentation to ensure all required evidence is captured:

| # | Exam Step / Requirement | Evidence / What to Show | Screenshot Checklist |
| :---: | :--- | :--- | :---: |
| **1** | **Dataset Inspection & Preprocessing** | Code in `model/preprocessor.py` showing invalid price/discount filtering and median imputation. | `[ ]` Code view of `engineer_features()` |
| **2** | **Feature Engineering** | Formulas for `Delivery_Delay_Days` and `Customer_Return_Frequency`. | `[ ]` Terminal output showing calculation verification |
| **3** | **Model Training & Evaluation** | Output of `python model/train.py` showing exact metrics for Logistic Regression & Random Forest. | `[ ]` Terminal screenshot of `model/metrics.json` |
| **4** | **Model Artifacts** | `model/best_model.joblib` and `model/metrics.json` generated in project tree. | `[ ]` VS Code explorer / directory listing |
| **5** | **FastAPI Implementation** | `app/main.py` code with `/health` and `/predict` endpoints. | `[ ]` Browser screenshot of Swagger UI (`/docs`) |
| **6** | **Automated Tests** | Execution of `python -m pytest -v` showing all 11 tests passing green. | `[ ]` Terminal screenshot of pytest summary |
| **7** | **Code Validation** | Execution of `flake8` and `python -m compileall`. | `[ ]` Terminal screenshot of 0 syntax errors |
| **8** | **Dockerfile & .dockerignore** | Clean multi-stage Dockerfile and exclusion configuration. | `[ ]` Code view of `Dockerfile` |
| **9** | **GitHub Actions Workflow File** | `.github/workflows/ci-cd.yml` with test, build, and publish stages. | `[ ]` Code view of `ci-cd.yml` |
| **10**| **GitHub Actions Pipeline Success** | GitHub Actions execution page showing green checkmarks on both jobs. | `[ ]` GitHub Actions run summary page |
| **11**| **GHCR Package Publication** | Repository Packages section showing published container image tag `latest`. | `[ ]` GitHub Packages registry page |
| **12**| **Live Container API Testing** | Running container locally and testing `/predict` with curl / Postman. | `[ ]` Response JSON with prediction & probability |
