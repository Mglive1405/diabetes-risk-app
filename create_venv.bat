@echo off
SET VENV_NAME=%1
IF "%VENV_NAME%"=="" SET VENV_NAME=venv

echo [1] Creating virtual environment: %VENV_NAME%
python -m venv %VENV_NAME%

echo [2] Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

echo [3] Upgrading pip inside venv...
python -m pip install --upgrade pip

echo [4] Installing common packages into venv...
python -m pip install ipython black flake8 requests pandas matplotlib

echo [✓] All done. Venv ready in folder: %VENV_NAME%
pause