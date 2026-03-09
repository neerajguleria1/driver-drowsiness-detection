import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Loading data...")
df = pd.read_csv('data/driving_scenario.csv')

print(f"Columns in CSV: {df.columns.tolist()}")

# Create missing features
df['Seatbelt'] = 1  # Assume seatbelt is on
df['HR'] = df['Heart_rate']  # Rename
df['speed_change'] = df['Speed'].diff().fillna(0).abs()  # Calculate speed change
df['prev_alertness'] = df['Alertness'].shift(1).fillna(df['Alertness'])  # Previous alertness

# Create target if not exists
if 'Drowsy' not in df.columns:
    # Create drowsy label based on alertness and fatigue
    df['Drowsy'] = ((df['Alertness'] < 0.5) | (df['Fatigue'] > 6)).astype(int)

print(f"Created features: {df.columns.tolist()}")

# Features and target
features = ['Speed', 'Alertness', 'Seatbelt', 'HR', 'Fatigue', 'speed_change', 'prev_alertness']
X = df[features]
y = df['Drowsy']

print(f"Training with {len(X)} samples...")

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X, y)

print(f"Model accuracy: {pipeline.score(X, y):.2%}")

# Save
joblib.dump(pipeline, 'models/final_driver_drowsiness_pipeline.pkl')
print("✅ Model retrained and saved successfully!")
print(f"✅ Model saved to: models/final_driver_drowsiness_pipeline.pkl")
