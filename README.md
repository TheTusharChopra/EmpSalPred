# Employee Salary Classification Project 💼

A comprehensive machine learning project that predicts whether an employee's annual income exceeds $50,000 based on demographic and employment characteristics using the Adult Census dataset.

## 🎯 Project Overview

This project implements a complete ML pipeline that:
- Processes and cleans employee data
- Compares 5 different machine learning algorithms
- Automatically selects the best performing model
- Deploys a user-friendly web application
- Provides public access via ngrok tunneling

## 📊 Dataset

The project uses the Adult Census dataset (`adult.csv`) containing:
- **15 features**: age, workclass, education, marital-status, occupation, relationship, race, gender, etc.
- **Target variable**: income (>50K or ≤50K)
- **Sample size**: ~32,000+ records after cleaning

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- 4GB+ RAM
- Internet connection

### Install Dependencies
```bash
pip install pandas scikit-learn streamlit matplotlib pyngrok joblib
```

### Ngrok Setup
1. Create account at [ngrok.com](https://ngrok.com)
2. Install ngrok
3. Set your auth token

## 🚀 Quick Start

### 1. Train the Model
```bash
python employee_salary_prediction.py
```
This will:
- Clean and preprocess the data
- Train 5 ML algorithms (LogisticRegression, RandomForest, KNN, SVM, GradientBoosting)
- Select and save the best model as `best_model.pkl`

### 2. Run the Web Application
```bash
streamlit run app.py
```

### 3. Create Public Access
```bash
python setup_ngrok.py
```
This generates a public URL accessible from anywhere.

## 📁 Project Structure

```
sal_pred/
├── adult.csv                      # Dataset
├── employee_salary_prediction.py  # Model training & evaluation
├── app.py                         # Streamlit web application
├── setup_ngrok.py                # Public deployment setup
├── best_model.pkl                # Saved trained model
└── README.md                     # Project documentation
```

## 🧠 Machine Learning Pipeline

### Data Preprocessing
- **Missing Value Handling**: Replace '?' with 'Others'
- **Outlier Removal**: Filter age (17-75) and education (5-16)
- **Feature Engineering**: Remove redundant columns
- **Label Encoding**: Convert categorical to numerical

### Model Comparison
| Algorithm | Description |
|-----------|-------------|
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble method with decision trees |
| K-Nearest Neighbors | Instance-based learning |
| Support Vector Machine | Kernel-based classification |
| Gradient Boosting | Advanced ensemble technique |

### Model Selection
- Automatic best model selection based on accuracy
- Model persistence using joblib
- Performance comparison visualization

## 🌐 Web Application Features

### Input Features
- **Age**: Slider (17-75)
- **Work Class**: Dropdown selection
- **Education Level**: Numerical input (5-16)
- **Marital Status**: Category selection
- **Occupation**: Job role options
- **Hours per Week**: Work schedule input

### Prediction Capabilities
- **Individual Predictions**: Real-time single employee classification
- **Batch Processing**: CSV file upload for multiple predictions
- **Results Download**: Export predictions as CSV

## 🔧 Technical Implementation

### Key Libraries Used
```python
# Data Processing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Web Application
import streamlit as st
from pyngrok import ngrok
```

### Model Pipeline
```python
# Pipeline with scaling and model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', best_model)
])
```

## 📈 Results & Performance

The system automatically selects the best performing algorithm and displays:
- Accuracy scores for all 5 models
- Classification reports with precision/recall
- Visual comparison charts
- Best model identification and saving

## 🌍 Deployment

### Local Deployment
1. Run Streamlit on localhost:8501
2. Access via web browser

### Public Access
1. Execute `setup_ngrok.py`
2. Get public URL (e.g., `https://abc123.ngrok.io`)
3. Share URL for remote access

### Tunnel Management
```python
# Automatic tunnel setup
public_url = ngrok.connect(8501)
print(f"🌐 App available at: {public_url}")
```

## 🔍 Usage Examples

### Making Predictions
1. Open the web application
2. Input employee details via sidebar
3. Click "Predict Salary Class"
4. View results (>50K or ≤50K)

### Batch Processing
1. Prepare CSV file with employee data
2. Upload via file uploader
3. Download results with predictions

## 🛠️ Troubleshooting

### Common Issues
- **Model not found**: Run `employee_salary_prediction.py` first
- **Ngrok errors**: Check auth token and internet connection
- **Import errors**: Verify all dependencies installed

### Performance Tips
- Use at least 4GB RAM for optimal performance
- Keep ngrok tunnel script running for continuous access
- Restart Streamlit if predictions seem slow

## 📝 Future Enhancements

- [ ] Add more sophisticated feature engineering
- [ ] Implement deep learning models
- [ ] Add data visualization dashboard
- [ ] Include model explainability features
- [ ] Deploy on cloud platforms (AWS/Heroku)

## 👨‍💻 Developer

**Tushar Chopra**
- Project: Employee Salary Classification System
- Technology Stack: Python, Scikit-learn, Streamlit, Ngrok

## 📄 License

This project is open source and available under the MIT License.

---

**🚀 Ready to predict salaries? Follow the Quick Start guide and get your ML application running in minutes!**
