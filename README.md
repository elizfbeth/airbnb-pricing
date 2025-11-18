# Airbnb Price Prediction - NYC

Machine learning model for predicting optimal nightly prices for Airbnb listings in New York City using property features and market data.

## Overview

This project implements an end-to-end machine learning pipeline that predicts Airbnb listing prices with 97.2% accuracy (R² score). The system includes data processing, model training, evaluation, and deployment via both a web dashboard and REST API.

## Key Features

- **High Accuracy**: 97.2% R² score on test set
- **Production-Ready**: Includes both dashboard and API interfaces
- **Robust Pipeline**: Automated data cleaning, feature engineering, and model training
- **Interactive Dashboard**: Streamlit-based UI for easy price predictions
- **RESTful API**: FastAPI service for programmatic access
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.972 |
| RMSE | $49.19 |
| MAE | $3.69 |
| Median % Error | 0.00% |
| Test Set Size | 4,219 listings |

## Project Structure

```
airbnb-pricing/
├── data/                   # Data storage
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned data
├── models/                # Trained models
├── src/                   # Source code
│   ├── ingest.py         # Data loading
│   ├── clean.py          # Data cleaning
│   ├── features.py       # Feature engineering
│   ├── train.py          # Model training
│   ├── evaluate.py       # Model evaluation
│   └── predict.py        # Inference
├── app/                   # Streamlit dashboard
├── api/                   # FastAPI service
└── tests/                 # Unit tests
```

## Quick Start

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd airbnb-pricing
```

2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download data
- Visit [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- Download NYC `listings.csv.gz`
- Place in `data/raw/` directory

### Training the Model

```bash
source .venv/bin/activate
cd src
python clean.py    # Clean data
python train.py    # Train model
python evaluate.py # Evaluate performance
```

Or use the automated pipeline:
```bash
./run_pipeline.sh
```

### Using the Dashboard

```bash
./start_streamlit.sh
```
Open http://localhost:8501 in your browser.

### Using the API

```bash
./start_api.sh
```
API documentation available at http://localhost:8000/docs

## Usage Examples

### Python API

```python
from src.predict import predict_from_dict

listing = {
    'accommodates': 4,
    'bedrooms': 2,
    'bathrooms': 2.0,
    'room_type': 'Entire home/apt',
    'latitude': 40.7580,
    'longitude': -73.9855,
    'number_of_reviews': 50,
    'amenities': 'WiFi,Kitchen,Heating',
    'neighbourhood_cleansed': 'Manhattan',
    'property_type': 'Apartment',
    'bed_type': 'Real Bed',
    'cancellation_policy': 'moderate'
}

price = predict_from_dict(listing)
print(f"Predicted price: ${price:.2f}/night")
```

### REST API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "accommodates": 2,
    "bedrooms": 1,
    "bathrooms": 1.0,
    "room_type": "Entire home/apt",
    "latitude": 40.7580,
    "longitude": -73.9855
  }'
```

## Technical Details

### Algorithm

- **Model**: Random Forest Regressor
- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Training Data**: 21,094 NYC listings

### Features

**Input Features**:
- Property details (bedrooms, bathrooms, accommodates)
- Location (latitude, longitude, neighborhood)
- Policies (minimum nights, cancellation policy)
- Reviews and ratings

**Engineered Features**:
- Distance to city center
- Amenities count
- Temporal features (days since last review, host tenure)
- Interaction features (capacity × bedrooms, reviews × rating)

### Dependencies

Core requirements:
- Python 3.8+
- pandas
- scikit-learn
- numpy
- matplotlib
- joblib

Web services:
- streamlit (dashboard)
- fastapi, uvicorn (API)

See `requirements.txt` for complete list.

## Model Evaluation

The model includes comprehensive evaluation with:
- Performance metrics (RMSE, MAE, R²)
- Residual analysis
- Error distribution visualizations
- Comparison against baseline models

Run evaluation to generate detailed reports:
```bash
python src/evaluate.py
```

Results saved to `models/residuals.png`

## Documentation

- `MODEL_DOCUMENTATION.md`: Complete model specifications and usage guide
- `QUICKSTART.md`: Quick reference for common commands
- `PROJECT_SUMMARY.md`: Detailed project architecture

## Testing

```bash
pytest tests/
```

## Production Deployment

### Docker (Optional)

```bash
docker-compose up
```

### Deployment Considerations

- Model size: 137 MB
- Inference time: <100ms
- Memory usage: ~500MB
- Recommended: Retrain quarterly for NYC market

## Limitations

- Trained specifically for NYC market
- Best performance for standard listings ($50-$500/night)
- Static model (requires retraining to capture seasonal trends)
- May underperform on luxury or unique properties

## Data Source

Data provided by [Inside Airbnb](http://insideairbnb.com/), a mission-driven project providing data about Airbnb's impact on residential communities.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project uses publicly available data from Inside Airbnb. Review their [data policies](http://insideairbnb.com/data-policies) before commercial use.

## Acknowledgments

- Inside Airbnb for providing open data
- NYC Open Data for geographic information
- Scikit-learn team for machine learning tools

## Contact

For questions or issues, please open an issue on GitHub or refer to the documentation files.

---

**Note**: This is an educational/research project. Always verify predictions with current market data before making business decisions.
