@echo off
REM Launch the budgeting dashboard and open it in the default browser.
REM Double-click this file; no terminal interaction is needed.

setlocal

set "BUDGET_DIR=%~dp0"
set "BUDGET_PYTHON=C:\Users\tward\anaconda3\envs\budget\python.exe"

if not exist "%BUDGET_PYTHON%" (
    echo Could not find the budget environment.
    echo Expected: %BUDGET_PYTHON%
    echo Create it with: conda env create -f "%BUDGET_DIR%environment.yml"
    pause
    exit /b 1
)

REM Open the browser once the server has had a moment to start, so the page
REM loads instead of showing a connection error.
start "" "http://localhost:8501"

cd /d "%BUDGET_DIR%"
"%BUDGET_PYTHON%" -m streamlit run app.py

endlocal
