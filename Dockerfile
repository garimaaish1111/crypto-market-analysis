# Cryptocurrency Market Analysis System
#
# Build:  docker build -t crypto-analysis .
# Run:    docker run -p 8501:8501 crypto-analysis
#
# With a CoinGecko demo key (strongly recommended — a keyless cold start spends
# more than a minute in rate-limit backoff):
#   docker run -p 8501:8501 -e COINGECKO_API_KEY=your_key crypto-analysis
#
# Offline, for a demo on an unreliable network:
#   docker run -p 8501:8501 -e CRYPTO_MODE=simulated crypto-analysis

FROM python:3.12-slim

# Faster, quieter, and no .pyc files written into the image layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements are copied first so the dependency layer is cached and only
# rebuilds when the pins actually change, not on every source edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The on-disk response cache lives here; declaring it means a container restart
# does not force a fresh cold fetch through the rate limiter.
VOLUME ["/app/data/cache"]

EXPOSE 8501

# Streamlit's own health endpoint, so an orchestrator can tell "still fetching"
# from "dead".
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
