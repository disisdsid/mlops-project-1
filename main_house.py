# main_house.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("house_model_v1.pkl")

# -----------------------------
# Input Schema
# -----------------------------
class HouseInput(BaseModel):
    bhk: int
    sqft: float
    location: str
    furnishing: str
    parking: int
    gym: int
    distance_metro: float
    nearby_schools: int
    nearby_malls: int
    floor: int
    property_type: str

# -----------------------------
# Basic Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Smart Real Estate API Live"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# UI PAGE
# -----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h2>🏠 House Price Predictor</h2>

        <form action="/predict-ui" method="post">
            BHK: <input name="bhk"><br><br>
            Sqft: <input name="sqft"><br><br>
            Location: <input name="location"><br><br>
            Furnishing: <input name="furnishing"><br><br>
            Parking: <input name="parking"><br><br>
            Gym: <input name="gym"><br><br>

            Distance to Metro: <input name="distance_metro"><br><br>
            Nearby Schools: <input name="nearby_schools"><br><br>
            Nearby Malls: <input name="nearby_malls"><br><br>
            Floor: <input name="floor"><br><br>
            Property Type: <input name="property_type"><br><br>

            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

# -----------------------------
# UI Prediction
# -----------------------------
@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    bhk: int = Form(...),
    sqft: float = Form(...),
    location: str = Form(...),
    furnishing: str = Form(...),
    parking: int = Form(...),
    gym: int = Form(...),
    distance_metro: float = Form(...),
    nearby_schools: int = Form(...),
    nearby_malls: int = Form(...),
    floor: int = Form(...),
    property_type: str = Form(...)
):
    input_df = pd.DataFrame([{
        "bhk": bhk,
        "sqft": sqft,
        "location": location,
        "furnishing": furnishing,
        "parking": parking,
        "gym": gym,
        "distance_metro": distance_metro,
        "nearby_schools": nearby_schools,
        "nearby_malls": nearby_malls,
        "floor": floor,
        "property_type": property_type
    }])

    prediction = model.predict(input_df)[0]

    if location.lower() == "indiranagar":
        prediction *= 1.2

    return f"<h2>💰 Price: ₹ {round(prediction,2)} Lakhs</h2><a href='/ui'>Back</a>"

# -----------------------------
# API Prediction
# -----------------------------
@app.post("/predict")
def predict(data: HouseInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]

    if data.location.lower() == "indiranagar":
        prediction *= 1.2

    return {"price": round(prediction, 2)}