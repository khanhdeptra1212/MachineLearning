# Lifestyle and Health Risk Prediction

##  Project Overview

This project focuses on predicting lifestyle and health risk levels using machine learning techniques. The system analyzes individual lifestyle indicators such as age, BMI, sleep duration, smoking habits, exercise frequency, alcohol consumption, and sugar intake to estimate potential health risks.

The project integrates **data analysis**, **machine learning model training**, and a **Flask-based web application** for user-friendly prediction.

---

##  Objectives

* Analyze lifestyle and health-related data
* Build a machine learning model to predict health risk
* Deploy the trained model using a Flask web application
* Ensure clean project structure and reproducibility

---

##  Project Structure

```
lifestyle-and-health/
│
├── app/                     # Flask application
│   ├── app.py               # Main Flask app
│   └── templates/
│       └── index.html       # Web interface
│
├── data/                    # Dataset
│   └── sample_data.csv      # Sample dataset (full dataset excluded)
│
├── notebooks/               # Data analysis & model training
│   └── lifestyle_and_health.ipynb
│
├── model/                   # Trained model
│   └── xgboost_pipeline.pkl
│
├── report/                  # Project report
│   └── Report_Datamining.docx
│
├── README.md
└── .gitignore
```

---

##  Dataset

* Only a **sample dataset** is included in this repository.
* The **full dataset is excluded** to reduce repository size and for data management purposes.
* Sample data maintains the same structure and column names as the original dataset.

---

## 🧠 Machine Learning Model

* Algorithm: **XGBoost (Pipeline)**
* Input features:

  * Age
  * BMI
  * Sleep duration
  * Smoking status
  * Exercise frequency
  * Alcohol consumption
  * Sugar intake
* Output:

  * Health risk prediction
  * Prediction probability

The trained model is saved as a `.pkl` file and loaded directly in the Flask application.

---

## 🌐 Web Application

The Flask web app allows users to:

* Enter lifestyle information via a web form
* Receive real-time health risk predictions

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd lifestyle-and-health
```

### 2. (Optional) Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not available, install manually:)*

```bash
pip install flask pandas scikit-learn xgboost joblib
```

### 4. Run the Flask application

```bash
cd app
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

bash
cd app
python app.py

```
Then open your browser at:
```

[http://127.0.0.1:5000](http://127.0.0.1:5000)

```

---

## 🧪 Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- Flask
- Jupyter Notebook

---

## 📌 Notes
- Development environment files and raw datasets are excluded using `.gitignore`.
- The project follows best practices in structuring machine learning and Flask applications.

---

## 📄 Report
A detailed project report is available in the `report/` directory, describing methodology, model training, evaluation, and system implementation.

---

## ✅ Conclusion
This project demonstrates a complete machine learning workflow, from data analysis and model training to deployment through a web application. The structured approach ensures clarity, scalability, and reproducibility, making the system suitable for academic and practical applications.

```
