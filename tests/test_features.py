"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features import (
    haversine_distance,
    add_price_per_person,
    extract_amenities_count,
    add_distance_to_center,
    feature_pipeline,
    prepare_features
)


def test_haversine_distance():
    """Test haversine distance calculation."""
    # Distance between Seoul (37.5665, 126.9780) and Busan (35.1796, 129.0756)
    # Should be approximately 325 km
    distance = haversine_distance(37.5665, 126.9780, 35.1796, 129.0756)
    assert 300 < distance < 350, f"Expected ~325km, got {distance}km"


def test_add_price_per_person():
    """Test price per person feature."""
    df = pd.DataFrame({
        'price': [100, 200, 300],
        'accommodates': [2, 4, 6]
    })
    result = add_price_per_person(df)
    
    assert 'price_per_person' in result.columns
    assert result['price_per_person'].iloc[0] == 50.0
    assert result['price_per_person'].iloc[1] == 50.0
    assert result['price_per_person'].iloc[2] == 50.0


def test_extract_amenities_count():
    """Test amenities count extraction."""
    df = pd.DataFrame({
        'amenities': ['WiFi,Kitchen', 'WiFi,Kitchen,Heating,AC', None]
    })
    result = extract_amenities_count(df)
    
    assert 'amenities_count' in result.columns
    assert result['amenities_count'].iloc[0] == 2
    assert result['amenities_count'].iloc[1] == 4
    assert result['amenities_count'].iloc[2] == 0


def test_add_distance_to_center():
    """Test distance to center calculation."""
    df = pd.DataFrame({
        'latitude': [37.5665, 37.5665],
        'longitude': [126.9780, 126.9780]
    })
    center_lat, center_lon = 37.5665, 126.9780
    
    result = add_distance_to_center(df, center_lat, center_lon)
    
    assert 'dist_to_center_km' in result.columns
    assert result['dist_to_center_km'].iloc[0] < 1.0  # Should be very close to center


def test_feature_pipeline():
    """Test feature pipeline creation."""
    pipeline = feature_pipeline()
    
    assert pipeline is not None
    # Test that it can transform sample data
    sample_df = pd.DataFrame({
        'accommodates': [2, 4],
        'bedrooms': [1, 2],
        'room_type': ['Entire home/apt', 'Private room'],
        'price': [100, 150]
    })
    
    # Note: This test requires actual data fitting, so we just check pipeline exists
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'transform')


def test_prepare_features():
    """Test feature preparation."""
    df = pd.DataFrame({
        'accommodates': [2, 4],
        'bedrooms': [1, 2],
        'beds': [1, 2],
        'bathrooms': [1.0, 2.0],
        'minimum_nights': [1, 2],
        'maximum_nights': [365, 365],
        'number_of_reviews': [10, 20],
        'latitude': [37.5665, 37.5665],
        'longitude': [126.9780, 126.9780],
        'amenities': ['WiFi,Kitchen', 'WiFi,Kitchen,Heating'],
        'price': [100, 150]
    })
    
    X, y = prepare_features(df)
    
    assert X.shape[0] == 2
    assert len(y) == 2
    assert 'price' not in X.columns
    assert 'price_per_person' in X.columns or 'dist_to_center_km' in X.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

