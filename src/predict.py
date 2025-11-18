import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any


def load_model(model_path: str = None):
    if model_path is None:
        # Use absolute path from project root
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "rf_pipeline.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return joblib.load(model_path)


def predict_from_dict(payload: Dict[str, Any],
                     model_path: str = None,
                     center_lat: float = None,
                     center_lon: float = None) -> float:
    from features import (
        extract_amenities_count,
        add_distance_to_center, add_temporal_features,
        create_interaction_features
    )

    # Convert to dataframe
    df = pd.DataFrame([payload])

    # Fill in default values for columns expected by model but not provided by user
    # These columns won't be used by the preprocessing pipeline but model expects them
    # Note: Many boolean fields use 't'/'f' strings, not True/False
    default_columns = {
        'id': 0,
        'listing_url': 'N/A',
        'scrape_id': 0,
        'last_scraped': '2025-01-01',
        'source': 'prediction',
        'name': 'New Listing',
        'description': 'N/A',
        'neighborhood_overview': 'N/A',
        'picture_url': 'N/A',
        'host_id': 0,
        'host_url': 'N/A',
        'host_name': 'Host',
        'host_since': '2025-01-01',
        'host_location': 'N/A',
        'host_about': 'N/A',
        'host_response_time': 'within an hour',
        'host_response_rate': '100%',
        'host_acceptance_rate': '100%',
        'host_is_superhost': 'f',
        'host_thumbnail_url': 'N/A',
        'host_picture_url': 'N/A',
        'host_neighbourhood': 'N/A',
        'host_listings_count': 1.0,
        'host_total_listings_count': 1.0,
        'host_verifications': "['email']",
        'host_has_profile_pic': 't',
        'host_identity_verified': 'f',
        'neighbourhood': 'N/A',
        'neighbourhood_group_cleansed': 'N/A',
        'bathrooms_text': '1 bath',
        'minimum_minimum_nights': 1.0,
        'maximum_minimum_nights': 1.0,
        'minimum_maximum_nights': 365.0,
        'maximum_maximum_nights': 365.0,
        'minimum_nights_avg_ntm': 1.0,
        'maximum_nights_avg_ntm': 365.0,
        'calendar_updated': None,
        'has_availability': 't',
        'availability_30': 30,
        'availability_60': 60,
        'availability_90': 90,
        'availability_365': 365,
        'calendar_last_scraped': '2025-01-01',
        'number_of_reviews_ltm': 0,
        'number_of_reviews_l30d': 0,
        'availability_eoy': 365,
        'number_of_reviews_ly': 0,
        'estimated_occupancy_l365d': 0.0,
        'estimated_revenue_l365d': 0.0,
        'first_review': None,
        'last_review': None,
        'review_scores_rating': 5.0,
        'review_scores_accuracy': 5.0,
        'review_scores_cleanliness': 5.0,
        'review_scores_checkin': 5.0,
        'review_scores_communication': 5.0,
        'review_scores_location': 5.0,
        'review_scores_value': 5.0,
        'license': 'N/A',
        'instant_bookable': 'f',
        'calculated_host_listings_count': 1,
        'calculated_host_listings_count_entire_homes': 0,
        'calculated_host_listings_count_private_rooms': 0,
        'calculated_host_listings_count_shared_rooms': 0,
        'reviews_per_month': 0.0,
    }

    # Add missing columns with defaults
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val

    # Add engineered features
    # Note: Skip add_price_per_person() since we don't have price during prediction
    df = extract_amenities_count(df)
    df = add_temporal_features(df)
    df = create_interaction_features(df)

    # Add distance to center if lat/lon provided
    if 'latitude' in df.columns and 'longitude' in df.columns:
        if center_lat is None:
            center_lat = df['latitude'].median()
        if center_lon is None:
            center_lon = df['longitude'].median()
        df = add_distance_to_center(df, center_lat, center_lon)

    # Add price_per_person as NaN (will be imputed by pipeline)
    if 'price_per_person' not in df.columns:
        df['price_per_person'] = 0.0  # Default value

    # Remove price if present (it's the target)
    if 'price' in df.columns:
        df = df.drop(columns=['price'])

    # Load model and predict
    model = load_model(model_path)
    prediction = model.predict(df)[0]

    return float(prediction)


if __name__ == "__main__":
    # Example usage
    sample_payload = {
        'accommodates': 2,
        'bedrooms': 1,
        'beds': 1,
        'bathrooms': 1,
        'room_type': 'Entire home/apt',
        'minimum_nights': 1,
        'maximum_nights': 365,
        'number_of_reviews': 10,
        'latitude': 37.5665,
        'longitude': 126.9780,
        'amenities': 'WiFi,Kitchen,Heating',
        'neighbourhood_cleansed': 'Gangnam',
        'property_type': 'Apartment',
        'bed_type': 'Real Bed',
        'cancellation_policy': 'moderate'
    }
    
    try:
        pred = predict_from_dict(sample_payload)
        print(f" Predicted price: ${pred:.2f}")
    except Exception as e:
        print(f" Error: {e}")
        print(" Make sure model is trained first (run train.py)")

