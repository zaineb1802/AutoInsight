@echo off
echo Starting AutoInsight API server...
cd /d "%~dp0"

:: Activate the AutoInsight virtualenv
call %~dp0..\..\AutoInsight\venv\Scripts\activate.bat

:: Point Python to the AutoInsight package
set AUTOINSIGHT_ROOT=%~dp0..\..\AutoInsight
set PYTHONPATH=%AUTOINSIGHT_ROOT%;%PYTHONPATH%

:: Copy .env so LLM keys are available
if exist "%AUTOINSIGHT_ROOT%\.env" (
    copy /Y "%AUTOINSIGHT_ROOT%\.env" .env >nul
    echo [OK] .env loaded from AutoInsight root.
)

echo AUTOINSIGHT_ROOT=%AUTOINSIGHT_ROOT%
echo Starting API on http://localhost:8000 ...

pip install fastapi "uvicorn[standard]" python-multipart --quiet
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000