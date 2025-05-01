import streamlit as st
import pandas as pd
import pickle

# Load model and expected feature columns
model = pickle.load(open('best_xgboost_model.pkl', 'rb'))
expected_features = pickle.load(open('model_features.pkl', 'rb'))

# Title
st.title("Mental Health Predictor – Growing Stress")
st.write("This app predicts whether a person is experiencing **growing stress** based on behavior and background.")

# Sidebar inputs
st.sidebar.header('Enter your information')

def user_input_features():
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
    Country = st.sidebar.selectbox('Country', ('United States', 'India', 'Canada', 'Other'))
    Occupation = st.sidebar.selectbox('Occupation', ('Corporate', 'Housewife', 'Student', 'Other'))
    self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    family_history = st.sidebar.selectbox('Family History of Mental Health', ('Yes', 'No'))
    treatment = st.sidebar.selectbox('Sought Treatment Before', ('Yes', 'No'))
    Days_Indoors = st.sidebar.selectbox('Days Indoors', ('Less than 7 days', '1-14 days', '15-30 days', 'More than 30 days'))
    Changes_Habits = st.sidebar.selectbox('Changes in Habits', ('Yes', 'No', 'Maybe'))
    Mental_Health_History = st.sidebar.selectbox('Mental Health History', ('Yes', 'No', 'Maybe'))
    Mood_Swings = st.sidebar.selectbox('Mood Swings Level', ('Low', 'Medium', 'High'))
    Coping_Struggles = st.sidebar.selectbox('Coping Struggles', ('Yes', 'No'))
    Work_Interest = st.sidebar.selectbox('Decrease in Work Interest', ('Yes', 'No'))
    Social_Weakness = st.sidebar.selectbox('Social Weakness', ('Yes', 'No'))
    mental_health_interview = st.sidebar.selectbox('Willing for Interview', ('Yes', 'No', 'Maybe'))
    care_options = st.sidebar.selectbox('Access to Care Options', ('Yes', 'No', 'Not Sure'))

    # Manual one-hot encoding
    data = {
        'Gender_Female': 1 if Gender == 'Female' else 0,
        'Gender_Other': 1 if Gender == 'Other' else 0,
        'Country_India': 1 if Country == 'India' else 0,
        'Country_Other': 1 if Country == 'Other' else 0,
        'Occupation_Housewife': 1 if Occupation == 'Housewife' else 0,
        'Occupation_Student': 1 if Occupation == 'Student' else 0,
        'Occupation_Other': 1 if Occupation == 'Other' else 0,
        'self_employed_Yes': 1 if self_employed == 'Yes' else 0,
        'family_history_Yes': 1 if family_history == 'Yes' else 0,
        'treatment_Yes': 1 if treatment == 'Yes' else 0,
        'Days_Indoors_1-14 days': 1 if Days_Indoors == '1-14 days' else 0,
        'Days_Indoors_15-30 days': 1 if Days_Indoors == '15-30 days' else 0,
        'Days_Indoors_More than 30 days': 1 if Days_Indoors == 'More than 30 days' else 0,
        'Changes_Habits_No': 1 if Changes_Habits == 'No' else 0,
        'Changes_Habits_Yes': 1 if Changes_Habits == 'Yes' else 0,
        'Mental_Health_History_No': 1 if Mental_Health_History == 'No' else 0,
        'Mental_Health_History_Yes': 1 if Mental_Health_History == 'Yes' else 0,
        'Mood_Swings_Medium': 1 if Mood_Swings == 'Medium' else 0,
        'Mood_Swings_High': 1 if Mood_Swings == 'High' else 0,
        'Coping_Struggles_Yes': 1 if Coping_Struggles == 'Yes' else 0,
        'Work_Interest_Yes': 1 if Work_Interest == 'Yes' else 0,
        'Social_Weakness_Yes': 1 if Social_Weakness == 'Yes' else 0,
        'mental_health_interview_No': 1 if mental_health_interview == 'No' else 0,
        'mental_health_interview_Yes': 1 if mental_health_interview == 'Yes' else 0,
        'care_options_No': 1 if care_options == 'No' else 0,
        'care_options_Not sure': 1 if care_options == 'Not Sure' else 0
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Align input with expected model features
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Show user inputs
st.subheader("Your Input:")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

# Map output class to label
prediction_label = {
    0: 'No Stress Growth',
    1: 'Maybe Stress Growing',
    2: 'Yes Stress Growing'
}

# Display result
st.subheader("Prediction Result")
st.markdown(f"### 🧠 **{prediction_label[prediction[0]]}**")
