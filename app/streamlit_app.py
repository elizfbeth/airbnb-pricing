"""
Streamlit dashboard for Airbnb pricing prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys
import os
import subprocess

# Download Git LFS files if needed (for Streamlit Cloud deployment)
def ensure_lfs_files():
    """Ensure Git LFS files are downloaded."""
    project_root = Path(__file__).resolve().parent.parent
    
    # Check if we need to download LFS files
    # LFS pointer files are small (<200 bytes) and start with "version https://git-lfs"
    lfs_files = [
        project_root / "data" / "processed" / "processed.csv",
        project_root / "data" / "processed" / "X_test.csv",
        project_root / "data" / "processed" / "y_test.csv",
        project_root / "models" / "rf_pipeline.joblib"
    ]
    
    needs_download = False
    for file_path in lfs_files:
        if file_path.exists():
            # Check if it's an LFS pointer (small file starting with "version")
            try:
                with open(file_path, 'rb') as f:
                    first_bytes = f.read(50)
                    if b'version https://git-lfs' in first_bytes:
                        needs_download = True
                        break
            except:
                pass
    
    if needs_download:
        try:
            # Run git lfs pull to download actual files
            result = subprocess.run(
                ['git', 'lfs', 'pull'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                st.cache_data.clear()  # Clear cache after downloading
        except Exception as e:
            # If git lfs pull fails, continue anyway (might not be needed)
            pass

# Run LFS download check (only once per session)
if 'lfs_checked' not in st.session_state:
    ensure_lfs_files()
    st.session_state.lfs_checked = True

# Add parent directory to path (so we can import from src)
# Use resolve() to get absolute path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Force reload of predict module to get latest changes
import importlib
if 'predict' in sys.modules:
    import predict
    importlib.reload(predict)
    from predict import predict_from_dict
else:
    from predict import predict_from_dict

from features import prepare_features

# Page config
st.set_page_config(
    page_title="Airbnb Pricing Predictor",
    page_icon="",
    layout="wide"
)

# Title
st.title(" Airbnb Pricing Predictor — NYC")
st.markdown("Predict optimal nightly prices for Airbnb listings in New York City using machine learning")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "EDA", "Model Performance", "Predict Price"]
)

# Cache model loading
@st.cache_data
def load_model():
    """Load trained model."""
    # Use absolute path from project root
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "rf_pipeline.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)

@st.cache_data
def load_data():
    """Load processed data."""
    # Use absolute path from project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "processed.csv"
    if not data_path.exists():
        return None
    # Check if file is too small (might be LFS pointer that wasn't downloaded)
    if data_path.stat().st_size < 1000:  # LFS pointers are <200 bytes
        return None
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        # File might be corrupted or still downloading
        return None

# Home page
if page == "Home":
    st.header("Project Overview")
    st.markdown("""
    This project predicts optimal nightly prices for Airbnb listings using machine learning.
    
    **Features:**
    - Data ingestion and cleaning pipeline
    - Feature engineering (distance to center, amenities count, etc.)
    - Multiple model comparison (RandomForest, XGBoost, LightGBM)
    - Interactive price prediction
    - SHAP explainability

    **Dataset:** Airbnb listings data (New York City)
    """)
    
    # Dataset summary
    df = load_data()
    if df is not None:
        st.subheader("Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Listings", f"{len(df):,}")
        col2.metric("Features", df.shape[1])
        if 'price' in df.columns:
            col3.metric("Avg Price", f"${df['price'].mean():.2f}")
            col4.metric("Median Price", f"${df['price'].median():.2f}")
    else:
        st.warning(" Dataset not found. Please run data processing pipeline first.")

# EDA page
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    df = load_data()
    if df is None:
        st.error(" Data not found. Please run data processing pipeline first.")
    else:
        # Price distribution
        st.subheader("Price Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Price ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Price Distribution')
        axes[0].grid(True, alpha=0.3)
        
        df['price_log'] = np.log1p(df['price'])
        axes[1].hist(df['price_log'], bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Log(Price + 1)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Price Distribution (Log-Transformed)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Price by room type
        if 'room_type' in df.columns:
            st.subheader("Price by Room Type")
            room_type_price = df.groupby('room_type')['price'].agg(['mean', 'median', 'count'])
            st.dataframe(room_type_price)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='price', by='room_type', ax=ax)
            plt.title('Price Distribution by Room Type')
            plt.suptitle('')
            plt.xlabel('Room Type')
            plt.ylabel('Price ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance")
    
    model = load_model()
    if model is None:
        st.error(" Model not found. Please train model first (run train.py).")
    else:
        # Load test metrics if available
        project_root = Path(__file__).resolve().parent.parent
        metrics_path = project_root / "models" / "residuals.png"
        if metrics_path.exists():
            st.subheader("Evaluation Metrics")
            st.image(str(metrics_path), use_container_width=True)
        
        # Display metrics from evaluation
        st.subheader("Performance Metrics")

        # Try to load and calculate metrics
        project_root = Path(__file__).resolve().parent.parent
        X_test_path = project_root / "data" / "processed" / "X_test.csv"
        y_test_path = project_root / "data" / "processed" / "y_test.csv"

        if X_test_path.exists() and y_test_path.exists():
            # Check if files are LFS pointers (too small)
            if X_test_path.stat().st_size < 1000 or y_test_path.stat().st_size < 1000:
                st.warning("⚠️ Data files appear to be Git LFS pointers. Please ensure Git LFS files are downloaded.")
                st.info("If deploying on Streamlit Cloud, make sure `git-lfs` is in `packages.txt` and files are committed.")
            else:
                try:
                    # Load test data
                    X_test = pd.read_csv(X_test_path)
                    y_test = pd.read_csv(y_test_path).squeeze()

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    from sklearn.metrics import mean_absolute_error, r2_score
                    try:
                        from sklearn.metrics import root_mean_squared_error
                        rmse = root_mean_squared_error(y_test, y_pred)
                    except ImportError:
                        from sklearn.metrics import mean_squared_error
                        rmse = mean_squared_error(y_test, y_pred, squared=False)

                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Display in columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error - Average prediction error")
                    col2.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error - Median prediction error")
                    col3.metric("R² Score", f"{r2:.3f}", help="Coefficient of determination - 1.0 is perfect")

                    # Additional metrics
                    st.markdown("---")
                    st.markdown("### Detailed Metrics")

                    pct_error = np.abs((y_pred - y_test) / y_test) * 100
                    median_pct_error = np.median(pct_error)
                    mean_pct_error = np.mean(pct_error)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Median % Error", f"{median_pct_error:.2f}%")
                    col2.metric("Mean % Error", f"{mean_pct_error:.2f}%")
                    col3.metric("Test Samples", f"{len(y_test):,}")

                    # Interpretation
                    st.markdown("---")
                    st.markdown("### Interpretation")
                    if r2 > 0.9:
                        st.success(f" **Excellent Performance!** Your model explains {r2*100:.1f}% of the variance in prices.")
                    elif r2 > 0.7:
                        st.info(f" **Good Performance!** Your model explains {r2*100:.1f}% of the variance in prices.")
                    else:
                        st.warning(f" **Moderate Performance.** Your model explains {r2*100:.1f}% of the variance in prices.")

                    st.markdown(f"""
                    - **RMSE of ${rmse:.2f}** means predictions are typically within ±${rmse:.0f} of actual prices
                    - **Median error of {median_pct_error:.2f}%** means half of predictions are nearly perfect
                    - **Tested on {len(y_test):,} listings** from NYC
                    """)
                except Exception as e:
                    st.error(f"Error loading or processing test data: {str(e)}")
                    st.info("If files are Git LFS pointers, ensure they are downloaded.")
        else:
            st.markdown("""
            | Metric | Value |
            |--------|-------|
            | RMSE | [Run evaluation] |
            | MAE | [Run evaluation] |
            | R² | [Run evaluation] |
            """)
            st.info(" Run `python src/evaluate.py` to generate detailed metrics and plots.")

# Predict Price page
elif page == "Predict Price":
    st.header("Predict Listing Price")
    
    model = load_model()
    if model is None:
        st.error(" Model not found. Please train model first (run train.py).")
    else:
        st.sidebar.subheader("Input Features")
        
        # Feature inputs
        accommodates = st.sidebar.number_input(
            "Accommodates", min_value=1, max_value=16, value=2
        )
        bedrooms = st.sidebar.number_input(
            "Bedrooms", min_value=0, max_value=10, value=1
        )
        beds = st.sidebar.number_input(
            "Beds", min_value=0, max_value=10, value=1
        )
        bathrooms = st.sidebar.number_input(
            "Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5
        )
        
        room_type = st.sidebar.selectbox(
            "Room Type",
            ["Entire home/apt", "Private room", "Shared room"]
        )
        
        minimum_nights = st.sidebar.number_input(
            "Minimum Nights", min_value=1, max_value=365, value=1
        )
        maximum_nights = st.sidebar.number_input(
            "Maximum Nights", min_value=1, max_value=365, value=365
        )
        
        number_of_reviews = st.sidebar.number_input(
            "Number of Reviews", min_value=0, max_value=1000, value=10
        )
        
        latitude = st.sidebar.number_input(
            "Latitude", min_value=-90.0, max_value=90.0, value=40.7580, step=0.0001
        )
        longitude = st.sidebar.number_input(
            "Longitude", min_value=-180.0, max_value=180.0, value=-73.9855, step=0.0001
        )
        
        amenities = st.sidebar.text_input(
            "Amenities (comma-separated)", value="WiFi,Kitchen,Heating"
        )
        
        neighbourhood = st.sidebar.text_input(
            "Neighbourhood", value="Manhattan"
        )
        
        property_type = st.sidebar.text_input(
            "Property Type", value="Apartment"
        )
        
        bed_type = st.sidebar.selectbox(
            "Bed Type",
            ["Real Bed", "Futon", "Couch", "Airbed", "Pull-out Sofa"]
        )
        
        cancellation_policy = st.sidebar.selectbox(
            "Cancellation Policy",
            ["flexible", "moderate", "strict", "super_strict_30", "super_strict_60"]
        )
        
        # Predict button
        if st.sidebar.button("Predict Price", type="primary"):
            # Create payload
            payload = {
                'accommodates': accommodates,
                'bedrooms': bedrooms,
                'beds': beds,
                'bathrooms': bathrooms,
                'room_type': room_type,
                'minimum_nights': minimum_nights,
                'maximum_nights': maximum_nights,
                'number_of_reviews': number_of_reviews,
                'latitude': latitude,
                'longitude': longitude,
                'amenities': amenities,
                'neighbourhood_cleansed': neighbourhood,
                'property_type': property_type,
                'bed_type': bed_type,
                'cancellation_policy': cancellation_policy
            }
            
            try:
                # Predict
                prediction = predict_from_dict(payload)

                # Display result
                st.success(f" Predicted Price: **${prediction:.2f}** per night")

                # Show input summary
                st.subheader("Input Summary")
                st.json(payload)

                # Feature importance (if available)
                st.subheader("Feature Importance")
                st.info(" SHAP values can be added here for explainability. Install shap package and add SHAP waterfall plot.")

            except Exception as e:
                import traceback
                st.error(f" Prediction error: {str(e)}")
                with st.expander("Show full error details"):
                    st.code(traceback.format_exc())
                st.info(" If you see a 'price' error, try clearing cache: Click the  menu → Settings → Clear cache → Rerun")

if __name__ == "__main__":
    pass

