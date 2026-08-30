@echo off
cd /d "%~dp0"

REM ------------------------------------------------------------------
REM Optional CoinGecko demo key. Strongly recommended.
REM
REM Without a key, six coins take roughly 30-40 seconds to load cold
REM (public rate limit, throttled to avoid 429s). A free demo key drops
REM that to a few seconds and mostly removes the throttling altogether.
REM
REM Get one free, no card needed, about two minutes:
REM   https://www.coingecko.com/en/developers/dashboard
REM Paste ONLY the key into a new file named apikey.txt in this same
REM folder (next to run.bat), then run this script again.
REM ------------------------------------------------------------------
if exist "apikey.txt" (
    set /p COINGECKO_API_KEY=<apikey.txt
    echo CoinGecko key loaded from apikey.txt - fast path enabled.
) else (
    echo.
    echo No apikey.txt found - running keyless. Crypto data will still load,
    echo just slower on a cold start. For a free demo key ^(~2 minutes^):
    echo   1. https://www.coingecko.com/en/developers/dashboard
    echo   2. Save the key alone, as text, in a new file: apikey.txt
    echo   3. Re-run this script.
    echo.
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
