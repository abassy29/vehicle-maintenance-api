# main.py
# FastAPI application integrating classification and regression prediction endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------
# 1. Initialize FastAPI
# ---------------------------
app = FastAPI(
    title="Vehicle Maintenance Prediction API",
    description="Endpoints for classification (needs maintenance) and regression (days to next maintenance)",
    version="1.0"
)

# ---------------------------
# 2. Load models and pipelines
# ---------------------------
# Classification (needs maintenance: Yes/No)
clf_model = joblib.load("model/random_forest_model.joblib")
scaler     = joblib.load("model/scaler.joblib")
encoders   = joblib.load("model/label_encoders.joblib")

# Regression (days until next maintenance)
# reg_preprocessor = joblib.load("preprocessing_pipeline.joblib")
# reg_model        = joblib.load("model/rf_maintenance_model.joblib")

# ---------------------------
# 3. Request schemas
# ---------------------------
class ClassifyInput(BaseModel):
    Vehicle_Model: str
    Reported_Issues: int
    Vehicle_Age: int
    Fuel_Type: str
    Transmission_Type: str
    Engine_Size: float
    Odometer_Reading: float
    Last_Service_Date: str
    Tire_Condition: str
    Brake_Condition: str
    Battery_Status: str

# class RegressionInput(BaseModel):
#     vehicle_type: str = Field(..., example="Car")
#     make: str         = Field(..., example="Toyota")
#     driving_condition: str = Field(..., example="Urban")
#     avg_daily_km: float
#     vehicle_age: int
#     total_km: float            # raw odometer reading
#     km_since_oil: float
#     km_since_oil_filter: float
#     km_since_fuel_filter: float
#     km_since_fuel_pump: float
#     km_since_battery: float
#     km_since_spark_plugs: float
#     km_since_shock_absorbers: float
#     km_since_brake_fluid: float
#     km_since_brake_pads: float
#     km_since_tires: float
#     km_since_coolant: float
#     km_since_water_pump: float
#     km_since_radiator: float
#     km_since_transmission_fluid: float
#     days_since_oil_change: int
#     days_since_oil_filter_change: int
#     days_since_fuel_filter_change: int
#     days_since_fuel_pump_change: int
#     days_since_battery_change: int
#     days_since_spark_plugs_change: int
#     days_since_shock_absorbers_change: int
#     days_since_brake_fluid_change: int
#     days_since_brake_pads_change: int
#     days_since_tires_change: int
#     days_since_coolant_change: int
#     days_since_water_pump_change: int
#     days_since_radiator_change: int
#     days_since_transmission_fluid_change: int

# ---------------------------
# 4. Classification endpoint
# ---------------------------
@app.post("/classify")
def classify(data: ClassifyInput):
    try:
        df = pd.DataFrame([data.dict()])
        # Preprocess: date to days since
        df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
        df['Days_Since_Last_Service'] = (pd.Timestamp.today() - df['Last_Service_Date']).dt.days
        df = df.drop(columns=['Last_Service_Date'])
        # Encode with validation
        for col, le in encoders.items():
            if col in df.columns:
                val = df.at[0, col]
                if val not in le.classes_:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Value '{val}' for '{col}' is not recognized. Allowed: {list(le.classes_)}"
                    )
                df[col] = le.transform([val])
        # Scale numeric
        num_cols = df.select_dtypes(include=[float, int]).columns
        df[num_cols] = scaler.transform(df[num_cols])

        pred  = clf_model.predict(df)[0]
        label = "Yes" if pred else "No"
        return {"Need_Maintenance": bool(pred), "Prediction_Label": label}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#----------------------------
# many classification endpoints

from typing import List

@app.post("/classify_many")
def classify_bulk(data_list: List[ClassifyInput]):
    try:
        records = []
        for data in data_list:
            row = data.dict()
            row['Last_Service_Date'] = pd.to_datetime(row['Last_Service_Date'])
            row['Days_Since_Last_Service'] = (pd.Timestamp.today() - row['Last_Service_Date']).days
            row.pop('Last_Service_Date')
            records.append(row)

        df = pd.DataFrame(records)

        # Encode categorical with safe fallback (-1 for unknowns)
        for col, le in encoders.items():
            if col in df.columns:
                known_classes = set(le.classes_)
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)

        # Scale numeric
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = scaler.transform(df[num_cols])

        preds = clf_model.predict(df)
        results = [{"Need_Maintenance": bool(p), "Prediction_Label": "Yes" if p else "No"} for p in preds]

        return {"predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# 5. Regression endpoint
# ---------------------------
# @app.post("/predict_days")
# def predict_days(input: RegressionInput):
#     try:
#         # Turn the Pydantic dict into the proper pipeline input
#         data = input.dict()
#         # Map total_km → current_odometer (what the pipeline expects)
#         data['current_odometer'] = data.pop('total_km')
#         df = pd.DataFrame([data])

#         # Preprocess & predict
#         X_proc = reg_preprocessor.transform(df)
#         days   = reg_model.predict(X_proc)[0]
#         return {"predicted_days_to_next_maintenance": round(float(days), 2)}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



#------------------------------------
# . Driver report analysis endpoint
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_utils import preprocess


# Load NLP models
category_model = joblib.load("model/incident_classifier_lr_fullprep.pkl")
urgency_model  = joblib.load("model/incident_urgency_lr_fullprep.pkl")
vectorizer     = joblib.load("model/tfidf_vectorizer_fullprep.pkl")

# Preprocessing
stop_words = set(stopwords.words('english'))
stop_words.update(['car', 'vehicle'])
lemmatizer = WordNetLemmatizer()
slang_map = {
    "shakin": "shaking",
    "ride": "car",
    "engin": "engine",
    "knoking": "knocking"
}

def preprocess(text: str) -> str:
    text = text.lower()
    tokens = text.split()
    tokens = [slang_map.get(tok, tok) for tok in tokens]
    text = ' '.join(tokens)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)

class DriverReport(BaseModel):
    report_text: str

@app.post("/analyze_report")
def analyze_report(data: DriverReport):
    try:
        cleaned = preprocess(data.report_text)
        tfidf   = vectorizer.transform([cleaned])
        issue   = category_model.predict(tfidf)[0]
        urgency = urgency_model.predict(tfidf)[0]
        return {
            "predicted_issue": issue,
            "predicted_urgency": urgency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#----------------------------
# ---------------------------
# Driver urgency analysis chart endpoint
from fastapi.responses import StreamingResponse
from io import BytesIO

from typing import List

class UrgencyList(BaseModel):
    predictions: List[int]


@app.post("/urgency_chart", response_class=StreamingResponse)
def generate_chart(data: UrgencyList):
    """
    Accepts a list of predicted urgency values (0=Low, 1=Medium, 2=High),
    and returns a bar chart showing the distribution.
    """
    urgency_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    urgencies = [urgency_map.get(p, 'Unknown') for p in data.predictions]

    df = pd.DataFrame({'urgency': urgencies})
    counts = df['urgency'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=['#2ecc40', '#ffb347', '#ff4136'], width=0.6)

    ax.set_title('Reports Urgency Levels Count', fontsize=16, fontweight='bold')
    #ax.set_xlabel('Urgency Level', fontsize=13)
    ax.set_ylabel('Number of Reports', fontsize=13)
    ax.tick_params(axis='x', labelsize=12, rotation=0)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type='image/png')

# ---------------------------
# 6. Health check
# ---------------------------
@app.get("/")
def root():
    return {"message": "Vehicle Maintenance Prediction API is running"}


#uvicorn api.api:app --reload

#ngrok http 8000