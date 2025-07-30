 #!/bin/bash

# Retail Intelligence Platform Startup Script

echo "🚀 Starting Retail Intelligence Platform..."

# Check if virtual environment exists
if [ ! -d "retail_env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv retail_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source retail_env/Scripts/activate
else
    source retail_env/bin/activate
fi

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{uploads,processed,samples,exports}
mkdir -p backend/models/saved_models
mkdir -p logs

# Start the application
echo "✅ Starting application on http://127.0.0.1:8000"
cd backend
python main.py
