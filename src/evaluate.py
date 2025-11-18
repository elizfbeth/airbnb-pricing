import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # For older scikit-learn versions
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


def evaluate_model(model_path: str = None,
                  X_test_path: str = None,
                  y_test_path: str = None,
                  output_dir: str = None) -> dict:

    print(" Evaluating model...")

    # Use absolute paths from project root
    project_root = Path(__file__).resolve().parent.parent

    if model_path is None:
        model_path = project_root / "models" / "rf_pipeline.joblib"
    if X_test_path is None:
        X_test_path = project_root / "data" / "processed" / "X_test.csv"
    if y_test_path is None:
        y_test_path = project_root / "data" / "processed" / "y_test.csv"
    if output_dir is None:
        output_dir = project_root / "models"

    # Load model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    print(f"  Loaded test set: {X_test.shape[0]} samples")
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Percentage error
    pct_error = np.abs((y_pred - y_test) / y_test) * 100
    median_pct_error = np.median(pct_error)
    mean_pct_error = np.mean(pct_error)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Median % Error': median_pct_error,
        'Mean % Error': mean_pct_error
    }
    
    print("\n Evaluation Metrics:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Median % Error: {median_pct_error:.2f}%")
    print(f"  Mean % Error: {mean_pct_error:.2f}%")
    
    # Create residual plot
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Predicted Price ($)')
    axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_pred - y_test
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Percentage error distribution
    axes[1, 1].hist(pct_error, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=median_pct_error, color='r', linestyle='--', lw=2, 
                       label=f'Median: {median_pct_error:.1f}%')
    axes[1, 1].set_xlabel('Percentage Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Percentage Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    residuals_path = output_dir / "residuals.png"
    plt.savefig(residuals_path, dpi=150, bbox_inches='tight')
    print(f"\n Residual plots saved to {residuals_path}")
    
    plt.close()
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    
    print("\n" + "="*50)
    print(" Evaluation complete!")
    print("="*50)

