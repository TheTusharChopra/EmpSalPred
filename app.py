import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs matching your CSV structure
st.sidebar.header("Input Employee Details")

# Age
age = st.sidebar.slider("Age", 17, 75, 30)

# Workclass options from your data
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                    'Local-gov', 'State-gov', 'Others']
workclass = st.sidebar.selectbox("Work Class", workclass_options)

# Final weight - not needed for prediction, using median value
fnlwgt = 189778  # median value from data

# Education options from your data
education_options = ['Bachelors', 'Some-college', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                    'Assoc-voc', 'Masters', '9th', '7th-8th', '12th', 'Doctorate',
                    '5th-6th', '10th', '11th', '1st-4th', 'Preschool']
educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)

# Marital status options
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                         'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)

# Occupation options from your data
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                     'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
                     'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
                     'Protective-serv', 'Armed-Forces', 'Others']
occupation = st.sidebar.selectbox("Occupation", occupation_options)

# Relationship options
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
relationship = st.sidebar.selectbox("Relationship", relationship_options)

# Race options
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
race = st.sidebar.selectbox("Race", race_options)

# Gender
gender_options = ['Female', 'Male']
gender = st.sidebar.selectbox("Gender", gender_options)

# Capital gain and loss
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)

# Hours per week
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)

# Native country
native_country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                         'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                         'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines',
                         'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                         'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                         'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
                         'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                         'Peru', 'Hong', 'Holand-Netherlands']
native_country = st.sidebar.selectbox("Native Country", native_country_options)

# Create label encoders with the same mapping as training
def create_encoders():
    # These mappings should match your training data preprocessing
    encoders = {}
    
    # Workclass mapping
    workclass_mapping = {'Federal-gov': 0, 'Local-gov': 1, 'Others': 2, 'Private': 3, 
                        'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6}
    
    # Marital status mapping  
    marital_mapping = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
                      'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
    
    # Occupation mapping
    occupation_mapping = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3,
                         'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6,
                         'Other-service': 7, 'Others': 8, 'Priv-house-serv': 9, 'Prof-specialty': 10,
                         'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14}
    
    # Relationship mapping
    relationship_mapping = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2,
                           'Own-child': 3, 'Unmarried': 4, 'Wife': 5}
    
    # Race mapping
    race_mapping = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
    
    # Gender mapping
    gender_mapping = {'Female': 0, 'Male': 1}
    
    # Native country mapping (simplified)
    country_mapping = {country: i for i, country in enumerate(sorted(native_country_options))}
    
    return {
        'workclass': workclass_mapping,
        'marital-status': marital_mapping,
        'occupation': occupation_mapping,
        'relationship': relationship_mapping,
        'race': race_mapping,
        'gender': gender_mapping,
        'native-country': country_mapping
    }

encoders = create_encoders()

# Function to preprocess input
def preprocess_input(age, workclass, fnlwgt, educational_num, marital_status, occupation,
                    relationship, race, gender, capital_gain, capital_loss, 
                    hours_per_week, native_country):
    
    # Create input array in the correct order
    input_data = [
        age,
        encoders['workclass'].get(workclass, 0),
        fnlwgt,
        educational_num,
        encoders['marital-status'].get(marital_status, 0),
        encoders['occupation'].get(occupation, 0),
        encoders['relationship'].get(relationship, 0),
        encoders['race'].get(race, 0),
        encoders['gender'].get(gender, 0),
        capital_gain,
        capital_loss,
        hours_per_week,
        encoders['native-country'].get(native_country, 0)
    ]
    
    return np.array(input_data).reshape(1, -1)

# Build input DataFrame for display
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class", type="primary"):
    try:
        # Preprocess input for prediction
        processed_input = preprocess_input(age, workclass, fnlwgt, educational_num, 
                                         marital_status, occupation, relationship, 
                                         race, gender, capital_gain, capital_loss, 
                                         hours_per_week, native_country)
        
        # Make prediction
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input) if hasattr(model, 'predict_proba') else None
        
        # Display result
        if prediction[0] == '>50K':
            st.success(f"✅ Prediction: **{prediction[0]}** - High Income")
        else:
            st.info(f"📊 Prediction: **{prediction[0]}** - Low Income")
            
        # Show probability if available
        if prediction_proba is not None:
            prob_low = prediction_proba[0][0] * 100
            prob_high = prediction_proba[0][1] * 100
            st.write(f"**Confidence:** ≤50K: {prob_low:.1f}% | >50K: {prob_high:.1f}%")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())
        
        # Process batch data (this would need similar preprocessing)
        st.warning("Batch prediction requires the same preprocessing as individual predictions. Ensure your CSV has all required columns.")
        
        if st.button("Process Batch Predictions"):
            # Note: This is a simplified version - you'd need to apply the same preprocessing
            # to the entire batch dataset before prediction
            st.info("Batch processing would apply the same preprocessing steps to your entire dataset.")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Model information
st.markdown("---")
st.markdown("#### ℹ️ Model Information")
st.info("This model predicts income based on demographic and work-related features from the Adult Census dataset.")