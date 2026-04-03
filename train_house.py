# train_house.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Dataset
data = {
    "bhk": [1, 2, 3, 2, 4, 3, 1, 2],
    "sqft": [600, 1200, 1800, 1000, 2500, 2000, 550, 1300],
    "location": ["Whitefield", "Indiranagar", "KR PURAM", "BTM", "Hebbal", "SarjaPur", "Hoskote", "Kormangla"],
    "furnishing": ["semi", "full", "semi", "unfurnished", "full", "semi", "unfurnished", "full"],
    "parking": [1, 1, 2, 0, 2, 1, 0, 1],
    "gym": [0, 1, 1, 0, 1, 1, 0, 1],
    "distance_metro": [1.2, 0.5, 2.0, 1.5, 0.3, 0.8, 2.5, 1.0],
    "nearby_schools": [2, 3, 1, 2, 4, 3, 1, 2],
    "nearby_malls": [1, 2, 1, 0, 3, 2, 0, 1],
    "floor": [2, 5, 7, 3, 10, 6, 1, 4],
    "property_type": ["new", "resale", "new", "resale", "new", "new", "resale", "new"],
    "price": [30, 80, 120, 60, 200, 150, 25, 90]
}

df = pd.DataFrame(data)

X = df.drop("price", axis=1)
y = df["price"]

categorical = ["location", "furnishing", "property_type"]
numerical = [
    "bhk", "sqft", "parking", "gym",
    "distance_metro", "nearby_schools",
    "nearby_malls", "floor"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numerical)
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "house_model_v1.pkl")

print("Model trained & saved as house_model_v1.pkl")