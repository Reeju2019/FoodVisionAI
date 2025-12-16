@echo off
echo ======================================================================
echo Starting FoodVisionAI API Server
echo ======================================================================
echo.

REM Check if GEMINI_API_KEY is set
if "%GEMINI_API_KEY%"=="" (
    echo WARNING: GEMINI_API_KEY is not set!
    echo.
    echo Please set it with:
    echo   set GEMINI_API_KEY=your_key_here
    echo.
    echo Or add it to a .env file
    echo.
    pause
)

echo Starting server on http://localhost:8000
echo.
echo Press CTRL+C to stop the server
echo.
echo ======================================================================
echo.

uvicorn foodvision_ai.api.main:app --reload --host 0.0.0.0 --port 8000

