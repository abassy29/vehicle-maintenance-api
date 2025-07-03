# preprocessing.py
# This script loads the maintenance dataset, performs feature engineering (including current_odometer), and preprocesses for modeling.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------
# 1. Load the dataset
# ---------------------------
# Update the path to your CSV file as needed
df = pd.read_csv('data/synthetic_maintenance_dataset_2000.csv')

# ---------------------------
# 2. Feature Engineering
# ---------------------------
# Add vehicle_age
df['vehicle_age'] = pd.Timestamp.now().year - df['year']

# Add current_odometer for clarity
# (alias for total_km)
df['current_odometer'] = df['total_km']

# Define parts list
parts = [
    'oil', 'oil_filter', 'fuel_filter', 'fuel_pump', 'battery', 'spark_plugs',
    'shock_absorbers', 'brake_fluid', 'brake_pads', 'tires', 'coolant',
    'water_pump', 'radiator', 'transmission_fluid'
]

# Compute km_since_<part> for each part
for part in parts:
    df[f'km_since_{part}'] = df['total_km'] - df[f'last_{part}_change_km']

# ---------------------------
# 3. Define features & target
# ---------------------------
target = 'target_days_to_next_maintenance'
# Include current_odometer in features
features = (
    ['vehicle_type', 'make', 'driving_condition', 'avg_daily_km', 'vehicle_age', 'current_odometer']
    + [f'km_since_{part}' for part in parts]
    + [f'days_since_{part}_change' for part in parts]
)
X = df[features]
y = df[target]

# ---------------------------
# 4. Split into train/test
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Preprocessing pipeline
# ---------------------------
numeric_features = (
    ['avg_daily_km', 'vehicle_age', 'current_odometer']
    + [f'km_since_{part}' for part in parts]
    + [f'days_since_{part}_change' for part in parts]
)
categorical_features = ['vehicle_type', 'make', 'driving_condition']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------------------
# 6. Save processed arrays and pipeline
# ---------------------------
# Save processed feature arrays
df_X_train = pd.DataFrame(
    X_train_processed,
    columns=[*numeric_features, *preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)]
)
df_X_test = pd.DataFrame(
    X_test_processed,
    columns=df_X_train.columns
)
df_X_train.to_csv('X_train_processed.csv', index=False)
df_X_test.to_csv('X_test_processed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Save the pipeline for future use
joblib.dump(preprocessor, 'preprocessing_pipeline.joblib')

print('Preprocessing complete. Files saved: X_train_processed.csv, X_test_processed.csv, y_train.csv, y_test.csv, preprocessing_pipeline.joblib')

# ---------------------------
# train_model.py
# This script loads preprocessed data, trains a Random Forest model, evaluates it, and saves the trained model.

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# 1. Load processed training and test sets
# ---------------------------
X_train = pd.read_csv('X_train_processed.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test_processed.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# ---------------------------
# 2. Initialize and train the model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------
# 3. Evaluate the model
# ---------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:\n"
      f"MAE: {mae:.2f} days\n"
      f"RMSE: {rmse:.2f} days\n"
      f"R² Score: {r2:.3f}")

# ---------------------------
# 4. Save the trained model
# ---------------------------
joblib.dump(model, 'model/rf_maintenance_model.joblib')
print('Trained model saved as rf_maintenance_model.joblib')
