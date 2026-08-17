@echo off
cd /d "%~dp0"

REM ------------------------------------------------------------------
REM Optional CoinGecko demo key.
REM Paste the key (and nothing else) into apikey.txt next to this file.
REM Only needed if you set CRYPTO_MODE = "live" in config.py.
REM ------------------------------------------------------------------
if exist "apikey.txt" (
    set /p COINGECKO_API_KEY=<apikey.txt
    echo CoinGecko key loaded from apikey.txt
)

REM ------------------------------------------------------------------
REM Locate a virtual environment.
REM
REM Preference order is deliberate: a shared environment one level ABOVE
REM this folder is checked first, because it survives replacing the
REM project folder. A venv stored inside the project is thrown away
REM every time a new version is extracted, which means reinstalling
REM every dependency for no reason.
REM
REM Recommended layout:
REM     ...\crypto\
REM         .venv\                    <- created once, never replaced
REM         crypto-market-analysis\   <- replace this folder freely
REM ------------------------------------------------------------------
set "VENV="
if defined CRYPTO_VENV if exist "%CRYPTO_VENV%\Scripts\activate.bat" set "VENV=%CRYPTO_VENV%"
if not defined VENV if exist "..\.venv\Scripts\activate.bat" set "VENV=..\.venv"
if not defined VENV if exist ".venv\Scripts\activate.bat" set "VENV=.venv"

if not defined VENV (
    echo.
    echo No virtual environment found. Creating a shared one at ..\.venv
    echo This happens once. Future versions of this project will reuse it.
    echo.
    python -m venv "..\.venv"
    if errorlevel 1 (
        echo.
        echo Could not create the environment. Is Python on your PATH?
        pause
        exit /b 1
    )
    call "..\.venv\Scripts\activate.bat"
    echo Installing dependencies, this takes a minute...
    pip install -r requirements.txt
    goto :launch
)

echo Using environment: %VENV%
call "%VENV%\Scripts\activate.bat"

REM Cheap self-heal: if a package is missing (new requirement, or a
REM half-finished install), top it up rather than failing at import.
python -c "import streamlit, plotly, statsmodels, pandas, numpy, scipy, requests" 2>nul
if errorlevel 1 (
    echo Some dependencies are missing. Installing...
    pip install -r requirements.txt
)

:launch
echo.
echo Starting dashboard. A browser tab should open automatically.
echo If it does not, open: http://localhost:8501
echo Press Ctrl+C in this window to stop.
echo.

streamlit run app.py
