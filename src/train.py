import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
from sklearn.dummy import DummyRegressor
import warnings
warnings.filterwarnings('ignore')

from features import prepare_features, feature_pipeline


def train_baseline(X_train, y_train, X_test, y_test):
    """Train baseline models for comparison."""
    print("\nTraining baseline models...")

    baselines = {
        'mean': DummyRegressor(strategy='mean'),
        'median': DummyRegressor(strategy='median')
    }

    results = {}
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"  {name.capitalize()}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

    return results


def train_random_forest(X_train, y_train, X_test, y_test,
                       numeric_features=None, categorical_features=None,
                       random_state=42, n_iter=20):
    """Train RandomForest with hyperparameter tuning."""
    print("\nTraining RandomForest model...")

    preprocessor = feature_pipeline(numeric_features, categorical_features)

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=random_state, n_jobs=-1))
    ])

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [8, 12, 20, None],
        'model__min_samples_leaf': [1, 2, 4],
        'model__min_samples_split': [2, 5, 10]
    }

    print("  Performing hyperparameter search...")
    search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=n_iter, cv=5,
        scoring='neg_mean_squared_error',
        random_state=random_state, n_jobs=-1, verbose=1
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'best_params': search.best_params_
    }

    print(f"  Best model:")
    print(f"     RMSE: {rmse:.2f}")
    print(f"     MAE: {mae:.2f}")
    print(f"     R2: {r2:.3f}")
    print(f"     Best params: {search.best_params_}")

    return best_pipeline, metrics


def main():
    """Main training pipeline."""
    print("Starting model training pipeline...")

    processed_path = Path("data/processed/processed.csv")
    if not processed_path.exists():
        print("Error: Processed data not found. Run clean.py first.")
        return

    df = pd.read_csv(processed_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nPreparing features...")
    X, y = prepare_features(df)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    baseline_results = train_baseline(X_train, y_train, X_test, y_test)

    rf_pipeline, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "rf_pipeline.joblib"
    joblib.dump(rf_pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_frame().to_csv("data/processed/y_test.csv", index=False)
    print(f"Test set saved for evaluation")

    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print("\nBaseline Models:")
    for name, metrics in baseline_results.items():
        print(f"  {name.capitalize()}: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")

    print("\nRandomForest:")
    print(f"  RMSE: {rf_metrics['RMSE']:.2f}")
    print(f"  MAE: {rf_metrics['MAE']:.2f}")
    print(f"  R2: {rf_metrics['R2']:.3f}")
    print("="*50)


if __name__ == "__main__":
    main()
