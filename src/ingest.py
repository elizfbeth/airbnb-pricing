import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - use absolute path from project root
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"


def load_listings(path: Optional[Path] = None) -> pd.DataFrame:
    file_path = path or RAW_DIR / "listings.csv.gz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Listings file not found at {file_path}. "
            "This is a mandatory dataset. Please download from Inside Airbnb."
        )

    logger.info(f"Loading listings from {file_path}")
    df = pd.read_csv(file_path, compression='gzip', low_memory=False)

    logger.info(f"Loaded {len(df):,} listings with {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def load_calendar(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    file_path = path or RAW_DIR / "calendar.csv.gz"

    if not file_path.exists():
        logger.warning(
            f"Calendar file not found at {file_path}. "
            "This is optional but recommended for advanced features."
        )
        return None

    logger.info(f"Loading calendar from {file_path}")
    df = pd.read_csv(file_path, compression='gzip', low_memory=False)

    # Parse date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df):,} calendar entries with {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def load_reviews(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    file_path = path or RAW_DIR / "reviews.csv.gz"

    if not file_path.exists():
        logger.warning(
            f"Reviews file not found at {file_path}. "
            "This is optional and only needed for NLP/sentiment features."
        )
        return None

    logger.info(f"Loading reviews from {file_path}")
    df = pd.read_csv(file_path, compression='gzip', low_memory=False)

    # Parse date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df):,} reviews with {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    logger.info("Loading all available Airbnb datasets...")

    datasets = {}

    # Load mandatory listings dataset
    datasets['listings'] = load_listings()

    # Load optional calendar dataset
    calendar = load_calendar()
    if calendar is not None:
        datasets['calendar'] = calendar

    # Load optional reviews dataset
    reviews = load_reviews()
    if reviews is not None:
        datasets['reviews'] = reviews

    logger.info(f"Successfully loaded {len(datasets)} dataset(s): {', '.join(datasets.keys())}")

    return datasets


def validate_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    validation = {}

    # Validate listings
    if 'listings' in datasets:
        listings = datasets['listings']
        validation['listings'] = {
            'rows': len(listings),
            'columns': len(listings.columns),
            'has_price': 'price' in listings.columns,
            'has_location': all(col in listings.columns for col in ['latitude', 'longitude']),
            'has_room_type': 'room_type' in listings.columns,
            'missing_price_pct': listings['price'].isna().mean() * 100 if 'price' in listings.columns else None
        }

    # Validate calendar
    if 'calendar' in datasets:
        calendar = datasets['calendar']
        validation['calendar'] = {
            'rows': len(calendar),
            'columns': len(calendar.columns),
            'has_date': 'date' in calendar.columns,
            'has_listing_id': 'listing_id' in calendar.columns,
            'date_range': (calendar['date'].min(), calendar['date'].max()) if 'date' in calendar.columns else None
        }

    # Validate reviews
    if 'reviews' in datasets:
        reviews = datasets['reviews']
        validation['reviews'] = {
            'rows': len(reviews),
            'columns': len(reviews.columns),
            'has_date': 'date' in reviews.columns,
            'has_listing_id': 'listing_id' in reviews.columns,
            'has_comments': 'comments' in reviews.columns
        }

    return validation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and validate Airbnb datasets from data/raw/"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["listings", "calendar", "reviews", "all"],
        default="all",
        help="Which dataset to load (default: all)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks on loaded datasets"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Save first N rows as sample CSV files"
    )

    args = parser.parse_args()

    try:
        # Load datasets based on selection
        if args.dataset == "all":
            datasets = load_all_datasets()
        elif args.dataset == "listings":
            datasets = {"listings": load_listings()}
        elif args.dataset == "calendar":
            calendar = load_calendar()
            datasets = {"calendar": calendar} if calendar is not None else {}
        elif args.dataset == "reviews":
            reviews = load_reviews()
            datasets = {"reviews": reviews} if reviews is not None else {}

        # Print summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        for name, df in datasets.items():
            print(f"\n{name.upper()}:")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Columns: {', '.join(df.columns[:10].tolist())}")
            if len(df.columns) > 10:
                print(f"           ... and {len(df.columns) - 10} more")

        # Run validation if requested
        if args.validate:
            print("\n" + "="*60)
            print("VALIDATION RESULTS")
            print("="*60)
            validation = validate_datasets(datasets)
            for dataset_name, stats in validation.items():
                print(f"\n{dataset_name.upper()}:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

        # Save samples if requested
        if args.sample:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            print("\n" + "="*60)
            print(f"SAVING SAMPLES (first {args.sample} rows)")
            print("="*60)
            for name, df in datasets.items():
                sample_path = PROCESSED_DIR / f"{name}_sample.csv"
                df.head(args.sample).to_csv(sample_path, index=False)
                print(f"  Saved {name} sample to {sample_path}")

        print("\n" + "="*60)
        print("SUCCESS")
        print("="*60)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        print("\n" + "="*60)
        print("ERROR: Missing Dataset")
        print("="*60)
        print(f"\n{e}")
        print("\nTo download datasets, visit: http://insideairbnb.com/get-the-data.html")
        print("Place the downloaded files in: data/raw/")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)
