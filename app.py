import streamlit as st
import pandas as pd
import numpy as np
from model import train_model, predict_salary, get_model_metrics
from preprocess import preprocess_input
import plotly.express as px
import plotly.graph_objects as go

# Load CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Load dataset
data = pd.read_csv('data/salary_dataset.csv')

# Train Model
model, X, y = train_model(data)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>SalarySense AI: Smart Salary Predictor 💰</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict your salary with real-time insights and feature impact analysis.</p>", unsafe_allow_html=True)

st.sidebar.header('Enter Your Profile Details')
experience = st.sidebar.slider('Years of Experience', 0, 20, 1)
education = st.sidebar.selectbox('Education Level', ['Bachelor', 'Master', 'PhD'])
company = st.sidebar.selectbox('Company Type', ['Service', 'Product', 'Startup'])
city = st.sidebar.selectbox('City', ['Delhi', 'Mumbai', 'Bangalore'])

if st.sidebar.button('Predict Salary'):
    input_df = preprocess_input(experience, education, company, city)
    predicted_salary = predict_salary(model, input_df)

    st.markdown(f"<h2 style='color: #ff6347;'>Predicted Salary: ₹{predicted_salary[0]:,.2f}</h2>", unsafe_allow_html=True)

    # Plot using Plotly
    fig = px.scatter(data, x='Experience', y='Salary', color='Education', title='Experience vs Salary with Education Levels')
    fig.add_traces(go.Scatter(x=data['Experience'], y=model.predict(X), mode='lines', name='Regression Line', line=dict(color='red')))
    st.plotly_chart(fig)

    # Model Performance
    mae, mse, r2 = get_model_metrics(model, X, y)
    st.subheader('Model Performance Metrics')
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    st.success('Prediction complete! 🎉')
