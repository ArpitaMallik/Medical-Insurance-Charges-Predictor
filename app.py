import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import xgboost as xgb


app = FastAPI()

# with open("xgb_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

model = xgb.XGBRegressor()
model.load_model("xgb_model_json.json")

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

class InputData(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: str  
    sex: str 
    region: str


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_html():
    return FileResponse("index.html") 



@app.post("/predict")
def predict(data: InputData):
    try:
        smoker = 1 if data.smoker.lower() == "yes" else 0
        sex_male = 1 if data.sex.lower() == "male" else 0

        region_northwest = 1 if data.region.lower() == "northwest" else 0
        region_southeast = 1 if data.region.lower() == "southeast" else 0
        region_southwest = 1 if data.region.lower() == "southwest" else 0

        # Keep only the features that were scaled originally
        model_input = [[data.age, data.bmi, data.children, smoker]]  

        # Scale only the relevant features
        scaled_input = scaler.transform(model_input)

        # Append unscaled categorical variables manually
        final_input = np.concatenate([scaled_input, [[sex_male, region_northwest, region_southeast, region_southwest]]], axis=1)

        log_prediction = model.predict(final_input)
        actual_prediction = float(np.exp(log_prediction)[0])
        
        return {"predicted_charges": round(actual_prediction, 2)}

    except Exception as e:
        return {"error": str(e)}