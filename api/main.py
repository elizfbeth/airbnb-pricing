"""
FastAPI server for Airbnb pricing prediction API.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path (so we can import from src)
# Use resolve() to get absolute path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from predict import predict_from_dict

app = FastAPI(
    title="Airbnb Pricing Prediction API",
    description="API for predicting optimal nightly prices for Airbnb listings",
    version="1.0.0"
)


class ListingFeatures(BaseModel):
    """Schema for listing features."""
    accommodates: int = 2
    bedrooms: int = 1
    beds: int = 1
    bathrooms: float = 1.0
    room_type: str = "Entire home/apt"
    minimum_nights: int = 1
    maximum_nights: int = 365
    number_of_reviews: int = 0
    latitude: float = 37.5665
    longitude: float = 126.9780
    amenities: Optional[str] = "WiFi,Kitchen"
    neighbourhood_cleansed: Optional[str] = "Gangnam"
    property_type: Optional[str] = "Apartment"
    bed_type: Optional[str] = "Real Bed"
    cancellation_policy: Optional[str] = "moderate"


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Airbnb Pricing Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict listing price",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: ListingFeatures):
    """
    Predict price for an Airbnb listing.
    
    Args:
        payload: Listing features
        
    Returns:
        Dictionary with predicted price
    """
    try:
        # Convert Pydantic model to dict
        features_dict = payload.dict()
        
        # Predict
        predicted_price = predict_from_dict(features_dict)
        
        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "USD",
            "features": features_dict
        }
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found. Please ensure model is trained: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(payloads: list[ListingFeatures]):
    """
    Predict prices for multiple listings (batch prediction).
    
    Args:
        payloads: List of listing features
        
    Returns:
        List of predictions
    """
    results = []
    
    for payload in payloads:
        try:
            features_dict = payload.dict()
            predicted_price = predict_from_dict(features_dict)
            results.append({
                "predicted_price": round(predicted_price, 2),
                "features": features_dict
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "features": payload.dict()
            })
    
    return {
        "predictions": results,
        "total": len(results)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

