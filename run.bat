@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv venv
  call venv\Scripts\activate.bat
  pip install -r requirements-ml.txt
) else (
  call venv\Scripts\activate.bat
)
python serve.py