# FoodVisionAI Project Runner
# This script starts the API server and opens the browser

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🚀 FoodVisionAI Project Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env file exists
if (Test-Path ".env") {
    Write-Host "✅ Found .env file" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: .env file not found" -ForegroundColor Yellow
    Write-Host "   Make sure GEMINI_API_KEY is set" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting API server on http://localhost:8000" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Available URLs:" -ForegroundColor Cyan
Write-Host "   - Upload Page:  http://localhost:8000/api/v1/upload" -ForegroundColor White
Write-Host "   - API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host "   - Health Check: http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Wait 2 seconds
Start-Sleep -Seconds 2

# Start the server
uvicorn foodvision_ai.api.main:app --reload --host 0.0.0.0 --port 8000

