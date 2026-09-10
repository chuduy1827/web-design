from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="House Price Predictor")

#Task 1: Prediction function (required)

def predict_price(area: float, bedrooms: int, location: str) -> float:
    base_price = 500_000_000.0
    total = base_price + (15_000_000.0 * area) + (50_000_000.0 * bedrooms)
    
    loc = location.strip().lower()
    if loc == "hanoi":
        total *= 1.3
    elif loc == "hcmc":
        total *= 1.25
        
    return round(total / 1_000_000) * 1_000_000

@app.get("/predict")
def predict(area: float, bedrooms: int, location: str = "other"):
    price=predict_price(area, bedrooms, location)
    return{
        "area": area,
        "bedrooms": bedrooms,
        "location": location,
        "predict_price": price
    }



app.mount("/static", StaticFiles(directory="../frontend"), name="static")