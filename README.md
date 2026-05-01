# 🏠 Egypt Real Estate Appraiser

**AI-powered real estate price prediction system for the Egyptian market**

---

## 🎯 Project Overview

This project presents a **complete Machine Learning pipeline** for predicting real estate prices in Egypt.
It transforms raw, semi-structured data into meaningful insights and accurate predictions using advanced regression models.

### 🔹 Key Features

* Data preprocessing pipeline (cleaning + transformation)
* Feature engineering (location grouping, price normalization)
* Multiple ML models (Linear, Ridge, Random Forest, XGBoost)
* Hyperparameter tuning using GridSearchCV
* Interactive web application using Streamlit
* Production-ready structure with modular design

---

## 🧠 Software Architecture (Design Patterns)

This project applies **three main design patterns** to improve code quality and scalability:

### 1. Strategy Pattern

Each preprocessing step is implemented as a separate class (strategy), such as:

* PriceCleaning
* SizeExtraction
* OutlierRemoval

🔹 **Why used?**

* Makes preprocessing modular
* Easy to add/remove steps
* Improves maintainability

---

### 2. Pipeline Pattern

A pipeline class executes all preprocessing steps in sequence.

🔹 **Why used?**

* Organizes workflow clearly
* Separates logic into stages
* Simplifies debugging

---

### 3. Factory Pattern

A factory class is used to create preprocessing steps.

🔹 **Why used?**

* Centralizes object creation
* Makes system scalable
* Reduces code duplication

---

## 📊 Dataset

* **Source**: Egyptian real estate listings
* **Records**: ~16,000
* **Target**: Property price (EGP)

### Key Features:

* Size (sqm)
* Bedrooms
* Bathrooms
* Location
* Property Type
* Payment Method

---

## 🔄 Data Processing Pipeline

### Step 1: Cleaning

* Remove invalid prices
* Extract size using regex
* Extract bedrooms from text
* Handle missing values

### Step 2: Transformation

* Remove duplicates
* Remove outliers (IQR method)
* Group rare locations

### Step 3: Feature Engineering

* Location grouping
* Price per sqm
* Size categories
* Luxury flag

---

## 🤖 Machine Learning

### Models Used:

* Linear Regression
* Ridge Regression
* Random Forest
* Gradient Boosting
* XGBoost ✅ (Best Model)

### Best Model Performance:

* **R² Score**: 0.82
* **RMSE**: ~748,000 EGP
* **MAE**: ~542,000 EGP

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing

```bash
python src/preprocessing.py
```

### 3. Train model

```bash
python src/train_model.py
```

### 4. Run app

```bash
python -m streamlit run app/app.py
```

---

## 📁 Project Structure

```
project/
│
├── data/
├── src/
│   ├── preprocessing.py
│   └── train_model.py
│
├── models/
├── app/
├── reports/
└── README.md
```

---

## 📊 Results

* Cleaned dataset: ~13,800 records
* Features engineered for better prediction
* XGBoost achieved highest accuracy



