#!/bin/bash
# Setup script for Airbnb Pricing Predictor

echo "🏠 Setting up Airbnb Pricing Predictor..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed models

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your dataset to data/raw/listings.csv"
echo "2. Run: python src/ingest.py"
echo "3. Run: python src/clean.py"
echo "4. Run: python src/train.py"
echo "5. Run: streamlit run app/streamlit_app.py"
echo ""

