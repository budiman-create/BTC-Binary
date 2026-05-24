"""
Robinhood Crypto API client.

Uses the official Robinhood Crypto Trading API (trading.robinhood.com)
with API-key + Ed25519 private-key authentication — no username/password needed.

Credentials go in .env:
    ROBINHOOD_API_KEY      = <your API key>
    ROBINHOOD_PRIVATE_KEY  = <base64-encoded Ed25519 private key>

Live quote data comes from Robinhood.
Historical OHLCV (for technical analysis) falls back to yfinance (BTC-USD, ETH-USD …)
since the official Robinhood API does not expose OHLCV candles.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass

import pandas as pd
import requests
import yfinance as yf
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_der_private_key,
    load_pem_private_key,
)

from .market_state import Horizon, StockState
from .technical_analysis import technical_summary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://trading.robinhood.com"

# Map our Horizon enum to yfinance period strings for historical data
_HORIZON_TO_YF_PERIOD: dict[Horizon, tuple[str, str]] = {
    Horizon.DAY:     ("1mo",  "1h"),
    Horizon.WEEK:    ("3mo",  "1h"),
    Horizon.MONTH:   ("1y",   "1d"),
    Horizon.QUARTER: ("2y",   "1d"),
}

# Robinhood uses "BTC-USD" style symbols; yfinance uses "BTC-USD" too
def _yf_symbol(symbol: str) -> str:
    s = symbol.upper()
    if "-" not in s:
        return f"{s}-USD"
    return s


# ---------------------------------------------------------------------------
# Auth — Ed25519 signed requests
# ---------------------------------------------------------------------------

def _load_private_key(b64_key: str) -> Ed25519PrivateKey:
    """Load Ed25519 private key from base64-encoded seed (32 bytes) or PKCS#8 DER."""
    raw = base64.b64decode(b64_key.strip())
    if len(raw) == 32:
        # Raw Ed25519 seed — the format Robinhood generates
        return Ed25519PrivateKey.from_private_bytes(raw)
    # Fall back: DER-encoded PKCS#8
    return load_der_private_key(raw, password=None)  # type: ignore[return-value]


def _make_headers(
    api_key: str,
    private_key: Ed25519PrivateKey,
    method: str,
    path: str,          # base path only — NO query string
    body: str = "",
) -> dict[str, str]:
    """Build signed request headers for Robinhood Crypto API."""
    timestamp = str(int(time.time()))
    # Robinhood signs: api_key + timestamp + path + METHOD + body
    # path must be the base path WITHOUT query parameters
    message = f"{api_key}{timestamp}{path}{method.upper()}{body}"
    signature = private_key.sign(message.encode("utf-8"))
    return {
        "x-api-key":   api_key,
        "x-signature": base64.b64encode(signature).decode("utf-8"),
        "x-timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class RobinhoodCryptoClient:
    """Thin wrapper around the Robinhood Crypto Trading REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        private_key_b64: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ROBINHOOD_API_KEY", "")
        pk_b64 = private_key_b64 or os.environ.get("ROBINHOOD_PRIVATE_KEY", "")

        if not self.api_key or not pk_b64:
            raise EnvironmentError(
                "Set ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY in your .env file."
            )

        self._private_key = _load_private_key(pk_b64)

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        from urllib.parse import urlencode
        # Signature must cover the full path INCLUDING query string
        qs = ("?" + urlencode(params)) if params else ""
        full = path + qs
        headers = _make_headers(self.api_key, self._private_key, "GET", full)
        resp = requests.get(BASE_URL + full, headers=headers, timeout=10)
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} from Robinhood: {resp.text}", response=resp
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Market data  (correct base: /api/v1/crypto/marketdata/ — no underscore)
    # ------------------------------------------------------------------

    def get_best_bid_ask(self, symbol: str) -> dict:
        data = self._get(
            "/api/v1/crypto/marketdata/best_bid_ask/",
            params={"symbol": _yf_symbol(symbol)},
        )
        results = data.get("results", [])
        if not results:
            raise ValueError(f"No quote data returned for {symbol}")
        return results[0]

    def get_estimated_price(self, symbol: str, side: str = "bid", quantity: float = 1.0) -> dict:
        return self._get(
            "/api/v1/crypto/marketdata/estimated_price/",
            params={
                "symbol": _yf_symbol(symbol),
                "side": side,
                "quantity": str(quantity),
            },
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        data = self._get("/api/v1/crypto/trading/accounts/")
        accounts = data.get("results", [data])
        return accounts[0] if accounts else {}

    def get_holdings(self) -> list[dict]:
        data = self._get("/api/v1/crypto/trading/holdings/")
        return data.get("results", [])


# ---------------------------------------------------------------------------
# Raw crypto data bundle
# ---------------------------------------------------------------------------

@dataclass
class CryptoData:
    symbol: str
    current_price: float
    bid_price: float | None
    ask_price: float | None
    history: pd.DataFrame       # OHLCV from yfinance

    @property
    def spread_pct(self) -> float | None:
        if self.ask_price and self.bid_price and self.current_price:
            return (self.ask_price - self.bid_price) / self.current_price
        return None

    def extra_context(self) -> str:
        parts = ["Asset class: Crypto"]
        if self.spread_pct is not None:
            parts.append(f"Bid-ask spread: {self.spread_pct:.3%}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch(
    symbol: str,
    horizon: Horizon = Horizon.MONTH,
    client: RobinhoodCryptoClient | None = None,
) -> CryptoData:
    """
    Fetch live quote from Robinhood + historical candles from yfinance.
    Pass a pre-built client to reuse credentials across multiple calls.
    """
    if client is None:
        client = RobinhoodCryptoClient()

    symbol = symbol.upper()

    # Live quote from Robinhood
    quote = client.get_best_bid_ask(symbol)
    bid   = float(quote.get("bid_inclusive_of_sell_spread") or quote.get("bid_price", 0) or 0)
    ask   = float(quote.get("ask_inclusive_of_buy_spread") or quote.get("ask_price", 0) or 0)
    price = (bid + ask) / 2 if bid and ask else float(quote.get("price", 0))

    # Historical candles from yfinance (Robinhood API has no OHLCV endpoint)
    yf_sym = _yf_symbol(symbol)
    period, interval = _HORIZON_TO_YF_PERIOD[horizon]
    df = yf.download(yf_sym, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Could not fetch historical data for {yf_sym} via yfinance.")

    # yfinance sometimes returns MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return CryptoData(
        symbol=symbol,
        current_price=price,
        bid_price=bid or None,
        ask_price=ask or None,
        history=df[["Open", "High", "Low", "Close", "Volume"]].dropna(),
    )


# ---------------------------------------------------------------------------
# Build StockState
# ---------------------------------------------------------------------------

def build_state(
    data: CryptoData,
    horizon: Horizon = Horizon.MONTH,
    analyst_drift_nudge: float = 0.0,
) -> StockState:
    tech = technical_summary(data.history, data.current_price)
    return StockState(
        ticker=data.symbol,
        current_price=data.current_price,
        annual_drift=tech["annual_drift"] + analyst_drift_nudge,
        annual_volatility=tech["annual_volatility"],
        trend_regime=tech["trend_regime"],
        vol_regime=tech["vol_regime"],
        horizon=horizon,
        rsi=tech["rsi"],
        pct_from_200ma=tech["pct_from_200ma"],
    )


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def get_positions(client: RobinhoodCryptoClient | None = None) -> list[dict]:
    """Return current crypto holdings from the Robinhood account."""
    if client is None:
        client = RobinhoodCryptoClient()

    raw = client.get_holdings()
    positions = []
    for h in raw:
        qty = float(h.get("quantity", 0) or 0)
        if qty == 0:
            continue
        positions.append({
            "symbol":            h.get("asset_code", "?"),
            "quantity":          qty,
            "average_buy_price": float(h.get("average_buy_price", 0) or 0),
        })
    return positions
