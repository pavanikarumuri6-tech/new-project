from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- 1. Load and Fix Data ---
df = pd.read_csv('MentalDisorder.csv')
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# --- 2. Encoding ---
ordinal_map = {'Seldom': 0, 'Sometimes': 1, 'Usually': 2, 'Most-Often': 3}
for col in ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']:
    df[col] = df[col].map(ordinal_map)

binary_cols = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation', 
               'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down', 'Admit Mistakes', 'Overthinking']
df[binary_cols] = df[binary_cols].replace({'YES': 1, 'NO': 0})

le = LabelEncoder()
df['Type of Disorder'] = le.fit_transform(df['Type of Disorder'])

# --- 3. Training ---
X = df.drop(['Patient Number', 'Type of Disorder'], axis=1)
y = df['Type of Disorder']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # We list the form names in the EXACT order they appear in the CSV features
    form_keys = ['sadness', 'euphoric', 'exhausted', 'sleep', 'mood', 'suicidal', 
                 'anorxia', 'respect', 'explanation', 'aggressive', 'ignore', 
                 'nervous', 'mistakes', 'overthinking']
    
    input_features = [int(request.form[key]) for key in form_keys]
    
    prediction = model.predict([input_features])
    result = le.inverse_transform(prediction)[0]
    
    return render_template('index.html', prediction_text=f'Analysis Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)