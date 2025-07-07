import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(data):
    X = data[['Experience', 'Education', 'Company', 'City']]
    y = data['Salary']

    categorical_features = ['Education', 'Company', 'City']
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)
    return model, X, y

def predict_salary(model, input_df):
    return model.predict(input_df)

def get_model_metrics(model, X, y):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, mse, r2
