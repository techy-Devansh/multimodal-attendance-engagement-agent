@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo SkillWallet - Python dependency installer
echo Recommended Python: 3.14 (64-bit)
echo ==============================================
echo.

py -3.14 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.14 was not found.
  echo Install 64-bit Python 3.14, then run this file again.
  echo Download: https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv314\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.14 -m venv .venv314
  if errorlevel 1 (
    echo ERROR: Could not create the virtual environment.
    pause
    exit /b 1
  )
)

echo Upgrading pip...
".venv314\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed.
  pause
  exit /b 1
)

echo Installing SkillWallet dependencies...
".venv314\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed. Read the error above.
  pause
  exit /b 1
)

echo.
echo Installation finished.
echo To run your app later, use:
echo .venv314\Scripts\python.exe -m streamlit run app.py
echo.
pause
