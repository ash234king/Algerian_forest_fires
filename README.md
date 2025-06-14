# 🔥 Algerian Forest Fire Prediction using Machine Learning

This project predicts the **Fire Weather Index (FWI)** based on meteorological data from the Algerian Forest Fire dataset. It includes complete preprocessing, model training with regression algorithms, evaluation, and deployment using **Flask** and **AWS Elastic Beanstalk**. A **CI/CD pipeline** is also configured using **AWS CodePipeline**.

---

## 📊 Problem Statement

The goal is to build a regression model that can predict the Fire Weather Index (FWI), which quantifies the fire risk, based on various environmental and meteorological parameters. The project also includes feature selection, model tuning, and deployment for real-world input predictions.

---

## 🗃 Dataset

- **Source**: Algerian Forest Fire Dataset (https://www.kaggle.com/datasets/mbharti321/algerian-forest-fires-dataset-updatecsv)
- **Features**: Temperature, RH, WS, Rain, FFMC, DMC, DC, ISI, BUI, etc.
- **Target**: FWI (Fire Weather Index)
- **Cleaning**: Dropped date columns, encoded categorical target column (`Classes` → binary).

---

## ⚙️ Tech Stack

- **Languages**: Python
- **Libraries**: pandas, numpy, seaborn, matplotlib, scikit-learn
- **ML Models**: Linear Regression, Ridge, Lasso, ElasticNet, with CV
- **Deployment**: Flask, AWS Elastic Beanstalk
- **CI/CD**: AWS CodePipeline, GitHub, S3

---

## 📌 Key Steps

### ✅ 1. Data Preprocessing
- Dropped non-relevant columns (`day`, `month`, `year`)
- Label encoded `Classes` feature (Fire / Not Fire)
- Feature selection using correlation analysis (> 0.85 threshold)
- Feature scaling using `StandardScaler`

### ✅ 2. Model Training
- Trained multiple regression models:
  - Linear Regression
  - Ridge, Lasso, ElasticNet
  - RidgeCV, LassoCV, ElasticNetCV for hyperparameter tuning
- Evaluated with Mean Absolute Error and R² Score

### ✅ 3. Web App
- Built a **Flask** web application that:
  - Accepts user input via a form
  - Applies saved preprocessing (`scaler.pkl`)
  - Loads model (`ridge.pkl`)
  - Returns FWI prediction

### ✅ 4. Deployment
- Hosted on **AWS Elastic Beanstalk**
- Preprocessing and model serialized using `pickle`
- Created a CI/CD pipeline using:
  - AWS CodePipeline
  - GitHub source
  - Elastic Beanstalk as the deployment target



