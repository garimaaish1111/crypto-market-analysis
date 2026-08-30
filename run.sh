#!/usr/bin/env bash
cd "$(dirname "$0")"

# Optional CoinGecko demo key. Strongly recommended.
#
# Without a key, six coins take roughly 30-40 seconds to load cold (public
# rate limit, throttled to avoid 429s). A free demo key drops that to a few
# seconds and mostly removes the throttling altogether.
#
# Get one free, no card needed, about two minutes:
#   https://www.coingecko.com/en/developers/dashboard
# Paste ONLY the key into a new file named apikey.txt in this same folder
# (next to run.sh), then run this script again.
if [ -f "apikey.txt" ]; then
    export COINGECKO_API_KEY="$(head -n 1 apikey.txt | tr -d '[:space:]')"
    echo "CoinGecko key loaded from apikey.txt - fast path enabled."
else
    echo ""
    echo "No apikey.txt found - running keyless. Crypto data will still load,"
    echo "just slower on a cold start. For a free demo key (~2 minutes):"
    echo "  1. https://www.coingecko.com/en/developers/dashboard"
    echo "  2. Save the key alone, as text, in a new file: apikey.txt"
    echo "  3. Re-run this script."
    echo ""
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
