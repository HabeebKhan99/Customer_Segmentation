# 🛍️ Customer Segmentation System

A Machine Learning web application built with Python and Streamlit that predicts the customer segment based on customer information.

## 🚀 Features

- Customer Segmentation using K-Means Clustering
- Interactive Streamlit Web Application
- Real-time Customer Group Prediction
- Label Encoding for Categorical Data
- StandardScaler for Feature Scaling
- Clean and Professional User Interface

## 📂 Project Structure

Customer_Segmentation/

├── Dataset/
│ └── store_customers.csv

├── models/
│ ├── model.pkl
│ ├── scaler.pkl
│ ├── encoder.pkl
│ └── feature.pkl

├── App.py
├── Training.py
├── requirements.txt
└── README.md

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Encoding
4. Feature Scaling
5. Train K-Means Model
6. Save Trained Files
7. Predict Customer Group using Streamlit

## ▶️ Run The Project

Install all required libraries

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run App.py
```

If the above command doesn't work, use:

```bash
python -m streamlit run App.py
```

## 👨‍💻 Developed By

**Habeeb Khan**
