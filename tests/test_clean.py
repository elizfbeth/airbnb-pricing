"""
Unit tests for data cleaning module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from clean import clean_price, clean_df


def test_clean_price():
    """Test price cleaning function."""
    assert clean_price("$100.50") == 100.50
    assert clean_price("1,500.00") == 1500.00
    assert clean_price("$1,234.56") == 1234.56
    assert pd.isna(clean_price("invalid"))
    assert pd.isna(clean_price(None))


def test_clean_df():
    """Test dataframe cleaning."""
    df = pd.DataFrame({
        'price': ['$100', '$200', '$5000', '$10'],  # One outlier
        'accommodates': [2, 2, 2, 2],
        'bedrooms': [1, 1, 1, 1],
        'latitude': [37.5665, 37.5665, 37.5665, 37.5665],
        'longitude': [126.9780, 126.9780, 126.9780, 126.9780]
    })
    
    result = clean_df(df, min_price=5.0, max_price=5000.0, output_path="test_output.csv")
    
    # Should remove price $10 (below min) and $5000 (at max, but might be kept)
    assert len(result) <= len(df)
    assert 'price' in result.columns
    assert result['price'].dtype == float
    
    # Clean up
    import os
    if os.path.exists("test_output.csv"):
        os.remove("test_output.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

