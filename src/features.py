import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    R = 6371  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def add_price_per_person(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['price_per_person'] = df['price'] / df['accommodates'].clip(lower=1)
    return df


def extract_amenities_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'amenities' in df.columns:
        df['amenities_count'] = df['amenities'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
    else:
        df['amenities_count'] = 0
    return df


def add_distance_to_center(df: pd.DataFrame, 
                          center_lat: float, 
                          center_lon: float) -> pd.DataFrame:
    df = df.copy()
    df['dist_to_center_km'] = df.apply(
        lambda row: haversine_distance(
            row['latitude'], row['longitude'], center_lat, center_lon
        ),
        axis=1
    )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if 'last_review' in df.columns:
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df['days_since_last_review'] = (
            pd.Timestamp.now() - df['last_review']
        ).dt.days.fillna(365)
    
    if 'host_since' in df.columns:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['host_days_active'] = (
            pd.Timestamp.now() - df['host_since']
        ).dt.days.fillna(0)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:  
    df = df.copy()
    
    # Room capacity interactions
    if 'accommodates' in df.columns and 'bedrooms' in df.columns:
        df['accommodates_x_bedrooms'] = df['accommodates'] * df['bedrooms']
    
    # Review interactions
    if 'number_of_reviews' in df.columns and 'review_scores_rating' in df.columns:
        df['reviews_x_rating'] = df['number_of_reviews'] * df['review_scores_rating'].fillna(0)
    
    return df


def prepare_features(df: pd.DataFrame,
                   target_col: str = 'price',
                   center_lat: Optional[float] = None,
                   center_lon: Optional[float] = None) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    
    # Add engineered features
    df = add_price_per_person(df)
    df = extract_amenities_count(df)
    df = add_temporal_features(df)
    df = create_interaction_features(df)
    
    # Add distance to center
    if center_lat is None:
        center_lat = df['latitude'].median()
    if center_lon is None:
        center_lon = df['longitude'].median()
    
    df = add_distance_to_center(df, center_lat, center_lon)
    
    # Separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return X, y


def feature_pipeline(numeric_features: Optional[list] = None,
                   categorical_features: Optional[list] = None,
                   max_categories: int = 10) -> ColumnTransformer:
    # Default feature lists (will be auto-detected if None)
    # These are common Airbnb listing features
    default_numeric = [
        'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'minimum_nights', 'maximum_nights', 'number_of_reviews',
        'latitude', 'longitude', 'amenities_count',
        'dist_to_center_km', 'price_per_person',
        'days_since_last_review', 'host_days_active',
        'accommodates_x_bedrooms', 'reviews_x_rating'
    ]
    
    default_categorical = [
        'room_type', 'neighbourhood_cleansed', 'property_type',
        'bed_type', 'cancellation_policy'
    ]
    
    numeric_features = numeric_features or default_numeric
    categorical_features = categorical_features or default_categorical
    
    # Create transformers
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            max_categories=max_categories,
            sparse_output=False
        ))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop features not explicitly listed
    )
    
    return preprocessor


if __name__ == "__main__":
    # Example usage
    from clean import clean_df
    from ingest import load_listings

    # Load and clean data
    print("Loading raw data...")
    df_raw = load_listings()

    print("Cleaning data...")
    df_clean = clean_df(df_raw)

    # Prepare features
    print("Engineering features...")
    X, y = prepare_features(df_clean)

    print(f"\n✅ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"\nFeature columns:")
    print(X.columns.tolist()[:10])  # Show first 10
    print(f"\nTarget statistics:")
    print(y.describe())

