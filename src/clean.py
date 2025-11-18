import pandas as pd
import numpy as np
from pathlib import Path
import re


def clean_price(price_str: str) -> float:
    if pd.isna(price_str):
        return np.nan
    
    # Convert to string and remove $ and commas
    price_str = str(price_str)
    price_str = re.sub(r'[\$,]', '', price_str)
    
    try:
        return float(price_str)
    except ValueError:
        return np.nan


def clean_df(df: pd.DataFrame, 
             min_price: float = 5.0,
             max_price: float = 5000.0,
             output_path: str = "data/processed/processed.csv") -> pd.DataFrame:

    df = df.copy()
    
    print(f" Starting cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {initial_rows - len(df)} duplicate rows")
    
    # 2. Clean price column
    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price)
        print(f"  Converted price column to float")
    else:
        print("    Warning: 'price' column not found")
    
    # 3. Remove extreme price outliers
    if 'price' in df.columns:
        before_outlier = len(df)
        df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        removed = before_outlier - len(df)
        print(f"  Removed {removed} rows with prices outside [{min_price}, {max_price}]")
    
    # 4. Handle missing values in key columns
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # For categorical columns, fill with mode or 'unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
            df[col].fillna(mode_val, inplace=True)
    
    print(f"  Handled missing values")
    
    # 5. Convert date columns
    date_columns = ['last_review', 'host_since', 'first_review']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"  Converted {col} to datetime")
    
    # 6. Ensure required columns exist (add defaults if missing)
    required_cols = {
        'accommodates': 2,
        'bedrooms': 1,
        'beds': 1,
        'bathrooms': 1,
        'minimum_nights': 1,
        'maximum_nights': 365,
        'number_of_reviews': 0,
        'latitude': 0.0,
        'longitude': 0.0,
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
            print(f"    Added missing column '{col}' with default value {default_val}")
    
    print(f" Cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"   Missing values: {df.isnull().sum().sum()} total")
    
    if 'price' in df.columns:
        print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Save cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Saved cleaned data to {output_path}")
    
    return df


if __name__ == "__main__":
    from ingest import load_listings

    # Load raw data
    print("Loading raw data from data/raw/listings.csv.gz...")
    df_raw = load_listings()

    # Clean data
    df_clean = clean_df(df_raw)

    print("\n Sample of cleaned data:")
    print(df_clean.head())

