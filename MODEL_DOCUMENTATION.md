# Model Documentation

## Overview

This document provides comprehensive information about the Airbnb pricing prediction model, including training methodology, performance metrics, and deployment instructions.

## Dataset

**Source**: Inside Airbnb (New York City)
**URL**: http://insideairbnb.com/get-the-data.html

**Training Data**:
- Total listings: 21,094
- Features: 79 (after engineering)
- Train/test split: 80/20
- Test set size: 4,219 listings

**Required Files**:
- `listings.csv.gz` (mandatory)
- `calendar.csv.gz` (optional)
- `reviews.csv.gz` (optional)

## Data Processing Pipeline

### 1. Data Ingestion (`src/ingest.py`)
Loads raw Airbnb data from CSV files with validation and error handling.

### 2. Data Cleaning (`src/clean.py`)
- Handles missing values
- Converts data types
- Parses price strings and removes currency symbols
- Handles date columns
- Validates required columns

### 3. Feature Engineering (`src/features.py`)
Creates engineered features to improve model performance:

**Temporal Features**:
- `days_since_last_review`: Days between last review and data collection
- `host_days_active`: Days host has been on platform

**Spatial Features**:
- `dist_to_center_km`: Distance to city center (km)

**Derived Features**:
- `amenities_count`: Number of amenities
- `price_per_person`: Price divided by accommodates (training only)

**Interaction Features**:
- `accommodates_x_bedrooms`: Capacity interaction
- `reviews_x_rating`: Reviews multiplied by rating score

## Model Architecture

**Algorithm**: Random Forest Regressor

**Preprocessing Pipeline**:
1. **Numeric Features**: StandardScaler + SimpleImputer (median strategy)
2. **Categorical Features**: OneHotEncoder + SimpleImputer (most_frequent strategy)

**Hyperparameter Tuning**:
- Method: RandomizedSearchCV with 5-fold cross-validation
- Iterations: 20
- Scoring: Negative mean squared error

**Optimal Parameters**:
- n_estimators: 300
- max_depth: None (unlimited)
- min_samples_split: 2
- min_samples_leaf: 1

## Performance Metrics

### Test Set Results

**Primary Metrics**:
- **RMSE**: $49.19
- **MAE**: $3.69
- **R² Score**: 0.972

**Error Analysis**:
- Median % Error: 0.00%
- Mean % Error: 0.34%

### Interpretation

**R² = 0.972**:
The model explains 97.2% of the variance in pricing, indicating excellent predictive performance.

**RMSE = $49.19**:
On average, predictions deviate by $49 from actual prices. For a dataset with prices ranging from $10 to $5000+, this represents strong accuracy.

**MAE = $3.69**:
Half of all predictions are within $3.69 of actual prices, demonstrating the model's precision for typical listings.

### Baseline Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Mean Baseline | $295.87 | $156.74 | -0.000 |
| Median Baseline | $306.90 | $142.50 | -0.076 |
| **RandomForest** | **$49.19** | **$3.69** | **0.972** |

The RandomForest model achieves 6x better performance than simple baseline models.

## Model Usage

### Making Predictions

```python
from src.predict import predict_from_dict

payload = {
    'accommodates': 4,
    'bedrooms': 2,
    'bathrooms': 2.0,
    'room_type': 'Entire home/apt',
    'latitude': 40.7580,
    'longitude': -73.9855,
    'minimum_nights': 2,
    'maximum_nights': 365,
    'number_of_reviews': 50,
    'amenities': 'WiFi,Kitchen,Heating,Air conditioning',
    'neighbourhood_cleansed': 'Manhattan',
    'property_type': 'Apartment',
    'bed_type': 'Real Bed',
    'cancellation_policy': 'moderate'
}

predicted_price = predict_from_dict(payload)
print(f"Predicted price: ${predicted_price:.2f}/night")
```

### Required Fields

Minimum required fields for prediction:
- `accommodates`, `bedrooms`, `beds`, `bathrooms`
- `room_type` ('Entire home/apt', 'Private room', 'Shared room')
- `latitude`, `longitude`
- `minimum_nights`, `maximum_nights`
- `number_of_reviews`

Missing optional fields will be filled with sensible defaults.

## Deployment

### Streamlit Dashboard

Interactive web interface for price predictions and model exploration.

```bash
./start_streamlit.sh
# Open http://localhost:8501
```

**Features**:
- Home: Dataset overview
- EDA: Exploratory visualizations
- Model Performance: Metrics and residual plots
- Predict Price: Interactive prediction form

### FastAPI Service

RESTful API for programmatic access.

```bash
./start_api.sh
# API docs at http://localhost:8000/docs
```

**Endpoints**:
- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Single prediction
- `POST /predict/batch`: Batch predictions

### Example API Request

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

## Production Recommendations

### Model Strengths
- High accuracy (R² > 0.9)
- Low error rates
- Handles missing data gracefully
- Fast inference time

### Limitations
- May underperform on luxury listings (>$1000/night)
- Limited to NYC market (retraining needed for other cities)
- Static model (doesn't capture seasonal trends without retraining)
- No confidence intervals provided

### Best Practices

**When to Use**:
- Standard residential listings ($50-$500/night)
- Properties with typical amenities
- Established neighborhoods

**When to Exercise Caution**:
- Unique property types (boats, castles, etc.)
- New neighborhoods not in training data
- Major market disruptions (e.g., COVID-19, major events)
- Luxury properties (>$1000/night)

### Monitoring

Track these metrics in production:
1. **Prediction Distribution**: Ensure predictions stay within expected ranges
2. **Error Rates**: Monitor RMSE/MAE over time
3. **User Feedback**: Collect actual vs predicted comparisons
4. **Market Changes**: Retrain quarterly or when significant drift detected

### Retraining Schedule

Recommended retraining frequency:
- **Quarterly**: For stable markets
- **Monthly**: During high volatility periods
- **Ad-hoc**: After major events affecting housing prices

## Files and Directories

```
airbnb-pricing/
├── data/
│   ├── raw/                 # Input datasets
│   └── processed/           # Cleaned and processed data
├── models/
│   ├── rf_pipeline.joblib   # Trained model (137 MB)
│   └── residuals.png        # Evaluation plots
├── src/
│   ├── ingest.py           # Data loading
│   ├── clean.py            # Data cleaning
│   ├── features.py         # Feature engineering
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   └── predict.py          # Inference
├── app/
│   └── streamlit_app.py    # Dashboard
├── api/
│   └── main.py             # REST API
└── tests/                  # Unit tests
```

## Training Pipeline

### Full Pipeline Execution

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Load and clean data
cd src
python clean.py

# 3. Train model
python train.py

# 4. Evaluate model
python evaluate.py
```

### Quick Start

```bash
./run_pipeline.sh  # Runs entire pipeline
```

## Evaluation Outputs

The evaluation script generates:

**Console Output**:
- RMSE, MAE, R² scores
- Percentage error statistics

**Visual Output** (`models/residuals.png`):
1. Predicted vs Actual scatter plot
2. Residual plot
3. Residual distribution histogram
4. Percentage error distribution

## Technical Specifications

**Dependencies**:
- Python 3.8+
- pandas, numpy
- scikit-learn 1.0+
- matplotlib
- joblib
- streamlit (for dashboard)
- fastapi, uvicorn (for API)

**Model Size**: 137 MB
**Inference Time**: <100ms per prediction
**Memory Usage**: ~500MB (model loaded)

## Version History

**v1.0** (Current)
- Initial release
- RandomForest with 97.2% R² accuracy
- Streamlit dashboard
- FastAPI service

## Support

For issues or questions:
1. Check `QUICKSTART.md` for common commands
2. Review `PROJECT_SUMMARY.md` for architecture details
3. Inspect test files in `tests/` for usage examples

## References

- Inside Airbnb: http://insideairbnb.com/
- Scikit-learn Documentation: https://scikit-learn.org/
- Random Forest Algorithm: Breiman, L. (2001). "Random Forests"

## License

This project uses publicly available data from Inside Airbnb. Please review their data policies before commercial use.
