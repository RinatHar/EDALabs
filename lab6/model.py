# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from eda import load_data

def preprocess_data(df):
    df = df.copy()
    categorical = ['Month', 'VisitorType']
    label_encoders = {}
    
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    return X, y, label_encoders

def train_model():
    df = load_data()
    X, y, label_encoders = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/shopper_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    joblib.dump(list(X.columns), 'model/feature_names.pkl')
    
    return accuracy

def predict(input_data):
    model = joblib.load('model/shopper_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    
    input_df = pd.DataFrame([input_data])
    
    for col in ['Month', 'VisitorType']:
        if col in input_df.columns:
            le = label_encoders[col]
            if input_df[col].iloc[0] in le.classes_:
                input_df[col] = le.transform([input_df[col].iloc[0]])
            else:
                input_df[col] = le.transform(['Other'])[0]
    
    if 'Weekend' in input_df.columns:
        input_df['Weekend'] = int(input_df['Weekend'])
    
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    
    return int(model.predict(input_scaled)[0])

if __name__ == '__main__':
    accuracy = train_model()
    print(f"Accuracy: {accuracy:.4f}")
    
    sample_data = {
        'Administrative': 3,
        'Administrative_Duration': 145.0,
        'Informational': 0,
        'Informational_Duration': 0.0,
        'ProductRelated': 20,
        'ProductRelated_Duration': 500.0,
        'BounceRates': 0.02,
        'ExitRates': 0.05,
        'PageValues': 10.5,
        'SpecialDay': 0.0,
        'Month': 'Nov',
        'OperatingSystems': 3,
        'Browser': 2,
        'Region': 1,
        'TrafficType': 2,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }
    
    try:
        prediction = predict(sample_data)
        print(f"Prediction: {'Purchase' if prediction == 1 else 'No Purchase'}")
    except:
        print("Model not trained yet")