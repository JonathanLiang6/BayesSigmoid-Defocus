@echo off
chcp 936 >nul
title Medical Analysis System

echo ============================================
echo   Defocus Bayesian Analysis System
echo ============================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "PYTHON_EXE=%PROJECT_DIR%venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist ".pytensor_cache" mkdir ".pytensor_cache"
if not exist "data\logs" mkdir "data\logs"
if not exist "data\results" mkdir "data\results"
if not exist "data\figures" mkdir "data\figures"

echo Select operation:
echo   1. Generate test data
echo   2. Run analysis
echo   3. Start Backend API (port 8000)
echo   4. Start Frontend Dev Server (port 5173)
echo   5. Start Both API + Frontend
echo   6. Run tests
echo   7. Run all (1-2 then start servers)
echo.

set /p choice=Enter choice (1-7): 

if "%choice%"=="1" goto generate_data
if "%choice%"=="2" goto run_analysis
if "%choice%"=="3" goto start_api
if "%choice%"=="4" goto start_frontend
if "%choice%"=="5" goto start_both
if "%choice%"=="6" goto run_tests
if "%choice%"=="7" goto run_all
echo Invalid choice
pause
exit /b 1

:generate_data
echo.
echo Generating test data...
"%PYTHON_EXE%" -m src.data_generator
echo Done. Data saved to data\raw_data.csv
echo.
pause
exit /b 0

:run_analysis
echo.
echo Running Bayesian analysis (5-15 min)...
"%PYTHON_EXE%" -m src.main
echo Done. Results in data\results\ and data\figures\
echo.
pause
exit /b 0

:start_api
echo.
echo Starting Backend API at http://localhost:8000
echo API Docs at http://localhost:8000/docs
echo Press Ctrl+C to stop
"%PYTHON_EXE%" -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
exit /b 0

:start_frontend
echo.
echo Starting Frontend at http://localhost:5173
echo Press Ctrl+C to stop
cd frontend
npm run dev
exit /b 0

:start_both
echo.
echo Starting Backend API in new window...
start "Backend API" "%PYTHON_EXE%" -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
timeout /t 3 /nobreak >nul
echo Starting Frontend in new window...
cd frontend
start "Frontend" npm run dev
echo Frontend: http://localhost:5173
echo.
echo Both servers started in separate windows.
echo Close those windows to stop the servers.
pause
exit /b 0

:run_tests
echo.
echo Running tests...
"%PYTHON_EXE%" -m pytest tests\test_data_processor.py -v
pause
exit /b 0

:run_all
echo.
echo === Step 1: Generate data ===
"%PYTHON_EXE%" -m src.data_generator
echo.
echo === Step 2: Run analysis (5-15 min) ===
"%PYTHON_EXE%" -m src.main
echo.
echo === Step 3: Start servers ===
start "Backend API" "%PYTHON_EXE%" -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
timeout /t 3 /nobreak >nul
cd frontend
start "Frontend" npm run dev
cd ..
echo.
echo Servers started:
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo.
pause
exit /b 0