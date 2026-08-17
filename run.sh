#!/usr/bin/env bash
cd "$(dirname "$0")"

# Optional CoinGecko demo key - see run.bat comments.
if [ -f "apikey.txt" ]; then
    export COINGECKO_API_KEY="$(head -n 1 apikey.txt | tr -d '[:space:]')"
    echo "CoinGecko key loaded from apikey.txt"
fi

# A shared environment one level above this folder is preferred, because it
# survives replacing the project folder. See run.bat for the full explanation.
VENV=""
if [ -n "$CRYPTO_VENV" ] && [ -f "$CRYPTO_VENV/bin/activate" ]; then
    VENV="$CRYPTO_VENV"
elif [ -f "../.venv/bin/activate" ]; then
    VENV="../.venv"
elif [ -f ".venv/bin/activate" ]; then
    VENV=".venv"
fi

if [ -z "$VENV" ]; then
    echo "No virtual environment found. Creating a shared one at ../.venv"
    echo "This happens once. Future versions of this project will reuse it."
    python3 -m venv "../.venv"
    VENV="../.venv"
    source "$VENV/bin/activate"
    pip install -r requirements.txt
else
    echo "Using environment: $VENV"
    source "$VENV/bin/activate"
    python -c "import streamlit, plotly, statsmodels, pandas, numpy, scipy, requests" 2>/dev/null \
        || { echo "Some dependencies are missing. Installing..."; pip install -r requirements.txt; }
fi

echo "Starting dashboard. If no browser opens: http://localhost:8501"
streamlit run app.py
