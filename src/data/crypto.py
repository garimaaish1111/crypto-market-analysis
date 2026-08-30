"""
Crypto price / volume / market-cap ingestion via the CoinGecko public API.

The free tier is aggressively rate limited — roughly 5 to 15 calls per minute
depending on load — and this project asks for six coins in a row. Firing them
back to back reliably earns a 429 on the last two or three, which then fall back
to simulated data while the first few stay live. That produces the worst possible
outcome: a risk table and correlation matrix silently mixing real and generated
assets.

Three things prevent that here. Requests are throttled to one every few seconds,
429 responses are retried with exponential backoff and honour ``Retry-After``,
and the source of every individual coin is reported back so the dashboard can
name exactly which assets are simulated rather than saying "mixed" and leaving
the reader to guess.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd
import requests

import config
from src.data import cache, sample_data

log = logging.getLogger(__name__)

_last_request_at = 0.0


def _throttle() -> None:
    """Space out live calls. Cache hits never reach this, so warm runs are instant."""
    global _last_request_at
    elapsed = time.time() - _last_request_at
    if elapsed < config.COINGECKO_MIN_INTERVAL:
        time.sleep(config.COINGECKO_MIN_INTERVAL - elapsed)
    _last_request_at = time.time()


def _headers() -> dict[str, str]:
    headers = {"accept": "application/json"}
    if config.API_KEY:
        # Demo and Pro keys use different header names and different hosts.
        # Sending a Pro header to the public host is silently ignored, which is
        # why an apparently valid key can still get rate limited.
        key_header = "x-cg-pro-api-key" if config.COINGECKO_PLAN == "pro" else "x-cg-demo-api-key"
        headers[key_header] = config.API_KEY
    return headers


class CoinGeckoError(RuntimeError):
    """A CoinGecko request that failed for a reason worth showing the user."""


def _get_with_retry(url: str, params: dict) -> requests.Response:
    """
    GET with backoff. Raises ``CoinGeckoError`` if every attempt is exhausted.

    Retries cover the two things that are worth retrying: rate limiting (429)
    and transient server or network faults (5xx, timeouts, DNS). A 401 or 403
    means the key is wrong or the endpoint is not on this plan — retrying that
    just burns fifteen seconds before failing anyway, so it raises immediately
    with a message that names the actual problem.
    """
    delay = config.COINGECKO_BACKOFF
    last_error: Exception | None = None

    for attempt in range(1, config.COINGECKO_MAX_ATTEMPTS + 1):
        _throttle()
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=config.REQUEST_TIMEOUT)
        except requests.Timeout as exc:
            last_error = CoinGeckoError(f"timed out after {config.REQUEST_TIMEOUT}s")
            log.warning("CoinGecko timeout (attempt %d/%d): %s", attempt, config.COINGECKO_MAX_ATTEMPTS, exc)
        except requests.ConnectionError as exc:
            last_error = CoinGeckoError("could not reach api.coingecko.com (no connection, DNS, or firewall)")
            log.warning("CoinGecko connection error (attempt %d/%d): %s", attempt, config.COINGECKO_MAX_ATTEMPTS, exc)
        except requests.RequestException as exc:
            last_error = CoinGeckoError(str(exc))
            log.warning("CoinGecko request error (attempt %d/%d): %s", attempt, config.COINGECKO_MAX_ATTEMPTS, exc)
        else:
            if resp.status_code == 429:
                # Honour the server's own guidance when it gives any.
                try:
                    wait = float(resp.headers.get("Retry-After", delay))
                except ValueError:
                    wait = delay
                wait = min(wait, config.COINGECKO_MAX_WAIT)
                log.warning(
                    "CoinGecko rate limited (attempt %d/%d); waiting %.0fs",
                    attempt, config.COINGECKO_MAX_ATTEMPTS, wait,
                )
                last_error = CoinGeckoError(
                    "rate limited (429). The keyless tier allows roughly 10-30 calls "
                    "a minute; a free demo key raises that to 100."
                )
                time.sleep(wait)
                delay *= 2
                continue

            if resp.status_code == 401:
                # Not transient. Retrying cannot help and only delays the answer.
                raise CoinGeckoError(
                    "401 — the API key was rejected. Check that COINGECKO_PLAN "
                    f"('{config.COINGECKO_PLAN}') matches the key: demo keys work against "
                    "api.coingecko.com, pro keys against pro-api.coingecko.com."
                )

            if resp.status_code == 403:
                # A 403 usually is not CoinGecko at all. Corporate proxies, VPNs
                # and Cloudflare all return it, and the message should say so
                # rather than sending someone off to re-check a key that is fine.
                raise CoinGeckoError(
                    "403 — access refused. This often comes from a proxy, VPN or "
                    "corporate firewall between you and CoinGecko rather than from "
                    "the API itself. Try the same URL in a browser to confirm."
                )

            if resp.status_code >= 500:
                last_error = CoinGeckoError(f"CoinGecko server error ({resp.status_code})")
                log.warning(
                    "CoinGecko %d (attempt %d/%d); retrying",
                    resp.status_code, attempt, config.COINGECKO_MAX_ATTEMPTS,
                )
            else:
                try:
                    resp.raise_for_status()
                except requests.HTTPError as exc:
                    raise CoinGeckoError(f"{resp.status_code} — {exc}") from exc
                return resp

        time.sleep(delay)
        delay *= 2

    raise last_error or CoinGeckoError("request failed")


def _fetch_market_chart(coin_id: str, days: int) -> pd.DataFrame:
    """Call CoinGecko and reshape prices/volumes/market-caps into a daily frame."""
    url = f"{config.coingecko_base()}/coins/{coin_id}/market_chart"
    payload = _get_with_retry(url, {"vs_currency": config.CURRENCY, "days": days}).json()

    def _series(field: str, name: str) -> pd.Series:
        arr = payload[field]
        return pd.Series(
            [v for _, v in arr],
            index=pd.to_datetime([t for t, _ in arr], unit="ms"),
            name=name,
        )

    df = pd.concat(
        [
            _series("prices", "price"),
            _series("total_volumes", "volume"),
            _series("market_caps", "market_cap"),
        ],
        axis=1,
    )
    # Collapse intraday points to one row per day.
    return df.resample("D").last().dropna()


def load_crypto(
    symbol: str, coin_id: str, days: int = config.DEFAULT_DAYS
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Return ``(dataframe, source, error)`` for one coin.

    ``source`` is one of:

    ``"live"``       fetched from CoinGecko
    ``"simulated"``  generated deliberately, because CRYPTO_MODE is "simulated"
    ``"sample"``     generated as an unplanned fallback after a live call failed

    ``error`` carries the reason for a ``"sample"`` result and is ``None``
    otherwise. Without it the cause of a fallback only ever reached the log,
    which a Streamlit user never sees — the dashboard could say a feed had
    dropped but not why, which is the one thing worth knowing.

    The last two sources produce identical data but mean different things, so
    the dashboard treats them differently: a chosen mode is stated calmly, an
    unexpected failure is flagged with its reason.
    """
    if config.CRYPTO_MODE != "live":
        # No network call, no exception to catch, nothing to log. The generator
        # is deterministic and fast, so there is nothing worth caching either.
        return sample_data.crypto_ohlcv(symbol, days), "simulated", None

    key = f"crypto:{coin_id}:{days}:{config.CURRENCY}"

    def _producer() -> tuple[pd.DataFrame, str, str | None]:
        try:
            return _fetch_market_chart(coin_id, days), "live", None
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we always recover
            reason = str(exc) or exc.__class__.__name__
            log.warning("CoinGecko fetch failed for %s (%s); using sample data", coin_id, reason)
            return sample_data.crypto_ohlcv(symbol, days), "sample", reason

    # Only a successful fetch is written to disk. Caching the fallback would
    # keep the dashboard on generated data for the full TTL even after the
    # network recovered.
    def _worth_caching(value: tuple) -> bool:
        return value[1] == "live"

    return cache.cached(key, _producer, should_cache=_worth_caching)


def load_all_crypto(
    days: int = config.DEFAULT_DAYS,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], str, dict[str, str], dict[str, str]]:
    """
    Load every configured coin.

    Returns the frames, an overall source label, a per-symbol source map so the
    dashboard can name which individual coins fell back rather than reporting an
    undifferentiated "mixed", and a per-symbol map of failure reasons.

    In simulated mode this loop never touches the network, so it returns
    immediately rather than spending the throttle interval on each coin.

    ``progress``, if given, is called as ``progress(done, total, symbol)`` once
    per coin, right after that coin finishes (live, simulated or fallen back to
    sample — the count advances either way, so the bar always reaches
    ``total``). This is the only change from omitting it: nothing about what is
    fetched, how it's throttled, or what is returned differs. A callback that
    raises is logged and ignored rather than allowed to abort a cold start over
    a UI-side bug.
    """
    frames: dict[str, pd.DataFrame] = {}
    per_symbol: dict[str, str] = {}
    errors: dict[str, str] = {}
    total = len(config.CRYPTO_ASSETS)

    for done, (coin_id, symbol) in enumerate(config.CRYPTO_ASSETS.items(), start=1):
        df, src, err = load_crypto(symbol, coin_id, days)
        frames[symbol] = df
        per_symbol[symbol] = src
        if err:
            errors[symbol] = err
        if progress is not None:
            try:
                progress(done, total, symbol)
            except Exception:  # noqa: BLE001 - a UI callback must never break ingestion
                log.warning("progress callback raised; ignoring", exc_info=True)

    sources = set(per_symbol.values())
    if sources == {"live"}:
        overall = "live"
    elif sources == {"simulated"}:
        overall = "simulated"
    elif "live" in sources:
        overall = "mixed"
    else:
        overall = "sample"
    return frames, overall, per_symbol, errors


def check_connection() -> tuple[bool, str]:
    """
    Ping CoinGecko once and report what happened, in words.

    This exists because "a feed dropped" is not a diagnosis. It bypasses the
    cache and the retry loop so the answer is about the network as it is right
    now, and it uses ``/ping`` rather than a coin so it costs one cheap call.
    """
    base = config.coingecko_base()
    key_state = (
        f"{config.COINGECKO_PLAN} key" if config.API_KEY else "keyless (no API key)"
    )
    try:
        resp = requests.get(
            f"{base}/ping", headers=_headers(), timeout=config.REQUEST_TIMEOUT
        )
    except requests.Timeout:
        return False, f"Timed out after {config.REQUEST_TIMEOUT}s reaching {base} — {key_state}."
    except requests.ConnectionError:
        return False, f"Could not reach {base} — no connection, DNS failure, or a firewall in the way."
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"

    if resp.status_code == 200:
        return True, f"Connected to {base} using {key_state}."
    if resp.status_code == 429:
        return False, (
            f"Rate limited (429) on {base} using {key_state}. "
            "A free demo key raises the ceiling from roughly 10-30 calls a minute to 100."
        )
    if resp.status_code == 401:
        return False, (
            f"401 from {base} using {key_state}. Demo keys must go to api.coingecko.com "
            "and Pro keys to pro-api.coingecko.com; check COINGECKO_PLAN matches your key."
        )
    if resp.status_code == 403:
        return False, (
            f"403 from {base} using {key_state}. This usually means a proxy, VPN or "
            "corporate firewall is blocking the request rather than CoinGecko refusing it. "
            "Open the URL in a browser to confirm."
        )
    return False, f"Unexpected {resp.status_code} from {base} using {key_state}."


def close_price_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-coin frames into a single price matrix (columns = symbols)."""
    prices = {sym: df["price"] for sym, df in frames.items()}
    return pd.DataFrame(prices).sort_index().ffill().dropna(how="all")
