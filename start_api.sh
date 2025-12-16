#!/bin/bash

echo "======================================================================"
echo "Starting FoodVisionAI API Server"
echo "======================================================================"
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY is not set!"
    echo ""
    echo "Please set it with:"
    echo "  export GEMINI_API_KEY=your_key_here"
    echo ""
    echo "Or add it to a .env file"
    echo ""
    read -p "Press Enter to continue anyway (API will fail on Stage 2)..."
fi

echo "Starting server on http://localhost:8000"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
echo "======================================================================"
echo ""

uvicorn foodvision_ai.api.main:app --reload --host 0.0.0.0 --port 8000

