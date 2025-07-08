import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import pandas as pd



def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Warranty_Expiry_Date',"Mileage","Maintenance_History","Owner_Type","Insurance_Premium","Service_History","Fuel_Efficiency", "Accident_History"])
    df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
    df['Days_Since_Last_Service'] = (pd.Timestamp.today() - df['Last_Service_Date']).dt.days
    df = df.drop(columns=['Last_Service_Date'])
    return df

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

def handle_missing_values(df):
    # Fill missing values with the mean for numerical columns
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing values with the mode for categorical columns
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df

def features_target(df):
    features = df.drop(columns=['Need_Maintenance'])
    target = df['Need_Maintenance']
    return features, target

def random_forest_classifier(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=25, criterion = "gini", min_samples_split=10)
    model.fit(x_train, y_train)
    return model


if __name__ == "__main__":
    df = load_csv("data/vehicle_maintenance_data.csv")
    df = remove_duplicates(df)
    df = handle_missing_values(df)


    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Need_Maintenance')
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save encoders and scaler
    joblib.dump(encoders, 'model/label_encoders.joblib')
    joblib.dump(scaler, 'model/scaler.joblib')

    # Split the data into features and target variable
    features, target = features_target(df)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    rf_model = random_forest_classifier(x_train, y_train)

    print(x_test.columns)
    print(x_test.info())
    print(x_test.head())

    # Make predictions on the test set
    y_pred = rf_model.predict(x_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


    # Save the model
    joblib.dump(rf_model, 'model/random_forest_model.joblib')





