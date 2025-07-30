echo "#!/bin/bash
# Quick Launch Commands for Retail Intelligence Platform

echo '🎯 Retail Intelligence Platform - Quick Launch'
echo '=============================================='

# Test platform
echo '1. Testing platform...'
python test_platform.py

# Generate sample data
echo '2. Generating sample data...'
python generate_sample_data.py

# Start platform
echo '3. Starting platform...'
echo 'Opening http://127.0.0.1:8000 in 3 seconds...'

# Start server in background
cd backend
python main.py &
SERVER_PID=\$!

# Wait and open browser
sleep 3

# Try to open browser (works on most systems)
if command -v python &> /dev/null; then
    python -c \"
import webbrowser
try:
    webbrowser.open('http://127.0.0.1:8000')
    print('✅ Browser opened!')
except:
    print('⚠️  Please manually open: http://127.0.0.1:8000')
\"
fi

echo '✅ Platform is running!'
echo '📖 Press Ctrl+C to stop the server'

# Wait for server process
wait \$SERVER_PID
" > quick_launch.sh

chmod +x quick_launch.sh