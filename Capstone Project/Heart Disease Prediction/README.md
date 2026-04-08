# Heart Disease Prediction using Machine Learning

## Overview
This project focuses on predicting the likelihood of heart disease using multiple machine learning (ML) classification models. The system leverages patient health parameters such as age, cholesterol, blood pressure, and heart rate to determine risk levels.

The project includes end-to-end implementation: data preprocessing, exploratory data analysis (EDA), feature selection, model training, evaluation, and comparison of multiple ML algorithms.

---

## Project Structure
```
├── Heart_Risk_Analytics.ipynb      # Main development notebook
├── proposal.pdf                   # Capstone proposal (if applicable)
├── project_report.pdf             # Final project report
├── heart_disease_dataset_UCI.csv  # Dataset (or link provided)
├── README.md                      # Project documentation
```

---

## Dataset
- **Source:** UCI Machine Learning Repository (via Kaggle)
- **File:** `heart_disease_dataset_UCI.csv`
- **Records:** Patient clinical data
- **Features:** 13 input variables + 1 target variable

### Key Features:
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Cholesterol (chol)
- Fasting blood sugar (fbs)
- Maximum heart rate (thalach)
- Exercise-induced angina (exang)
- Number of vessels (ca)
- Thalassemia (thal)

### Target:
- `0` → No heart disease  
- `1` → Presence of heart disease  

---

## Technologies and Libraries

### Programming Language
- Python 3.x

### Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- missingno
- scikit-learn
- xgboost
- lightgbm

---

## Methodology

### 1. Data Preprocessing
- Handling missing values
- Removing duplicates
- Outlier detection (IQR method, 5%–95% quantile filtering)
- Feature transformation (categorical encoding)

### 2. Exploratory Data Analysis (EDA)
- Feature distribution analysis
- Correlation heatmaps
- Feature vs target relationships

### 3. Feature Selection
- SelectKBest
- Random Forest feature importance
- XGBoost feature importance
- Principal Component Analysis (PCA)

### 4. Model Development
Models implemented:
- Logistic Regression (LGR)
- Linear Discriminant Analysis (LDA)
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

### 5. Model Optimization
- Randomized Search
- 5-fold cross-validation

### 6. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## Results
- Best performing models:
  - Logistic Regression
  - Linear Discriminant Analysis

- Achieved accuracy: ~87%–88%

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm missingno
```

### 2. Run Notebook
```bash
jupyter notebook Heart_Risk_Analytics.ipynb
```

### 3. Execute
Run all cells sequentially to reproduce results.

---

## Reproducibility Notes
- Keep dataset in the same directory as notebook
- Use same random seeds if defined
- Follow execution order

---

## References
- UCI Machine Learning Repository
- Kaggle Heart Disease Dataset
- Relevant research papers on heart disease prediction