from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphicalContextConfig:
    keltner_ema_period: int = 20
    atr_period: int = 14
    keltner_atr_multiplier: float = 2.0
    adx_period: int = 14
    market_structure_lookback: int = 20


class GraphicalContext:
    """Causal 15m graphical features for staged entry-confirmation research.

    Frozen v1 definitions:
    - Keltner: EMA20 +/- 2 * ATR14.
    - ADX/DMI: Wilder-style 14-period directional movement.
    - VWAP: UTC daily session VWAP using typical price * volume.
    - Market structure: close versus the prior 20-bar high/low; current bar is
      excluded from the reference range.

    No feature uses future bars. The class only creates features; trade rules
    remain the responsibility of the benchmark so each battery can be ablated.
    """

    def __init__(self, config: GraphicalContextConfig | None = None):
        self.config = config or GraphicalContextConfig()

    @staticmethod
    def _true_range(frame: pd.DataFrame) -> pd.Series:
        prev_close = frame["close"].shift(1)
        return pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - prev_close).abs(),
                (frame["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

    @staticmethod
    def _wilder(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "high", "low", "close", "volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"graphical context missing columns: {sorted(missing)}")

        c = frame.copy().sort_values("timestamp").reset_index(drop=True)
        high = pd.to_numeric(c["high"], errors="coerce")
        low = pd.to_numeric(c["low"], errors="coerce")
        close = pd.to_numeric(c["close"], errors="coerce")
        volume = pd.to_numeric(c["volume"], errors="coerce")

        # Keltner Channels.
        tr = self._true_range(c)
        atr = tr.rolling(self.config.atr_period, min_periods=self.config.atr_period).mean()
        middle = close.ewm(span=self.config.keltner_ema_period, adjust=False).mean()
        width = self.config.keltner_atr_multiplier * atr
        upper = middle + width
        lower = middle - width
        c["keltner_middle"] = middle
        c["keltner_upper"] = upper
        c["keltner_lower"] = lower
        c["keltner_width_pct"] = (upper - lower) / middle.replace(0.0, np.nan)
        c["keltner_position"] = (close - middle) / width.replace(0.0, np.nan)
        c["keltner_expanding"] = c["keltner_width_pct"] > c["keltner_width_pct"].shift(1)
        c["keltner_long_confirm"] = (close > middle) & c["keltner_expanding"]
        c["keltner_short_confirm"] = (close < middle) & c["keltner_expanding"]

        # ADX / DMI, using only current and prior bars.
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=c.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=c.index)
        atr_wilder = self._wilder(tr, self.config.adx_period)
        plus_di = 100.0 * self._wilder(plus_dm, self.config.adx_period) / atr_wilder.replace(0.0, np.nan)
        minus_di = 100.0 * self._wilder(minus_dm, self.config.adx_period) / atr_wilder.replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        adx = self._wilder(dx, self.config.adx_period)
        c["plus_di"] = plus_di
        c["minus_di"] = minus_di
        c["adx"] = adx
        c["adx_rising"] = adx > adx.shift(1)
        c["dmi_long_confirm"] = (plus_di > minus_di) & c["adx_rising"]
        c["dmi_short_confirm"] = (minus_di > plus_di) & c["adx_rising"]

        # Session VWAP reset at 00:00 UTC. This is causal within each UTC day.
        ts = pd.to_datetime(c["timestamp"], utc=True, errors="coerce")
        session = ts.dt.floor("D")
        typical = (high + low + close) / 3.0
        pv = typical * volume
        cum_pv = pv.groupby(session).cumsum()
        cum_vol = volume.groupby(session).cumsum()
        vwap = cum_pv / cum_vol.replace(0.0, np.nan)
        c["session_vwap"] = vwap
        c["vwap_distance_atr"] = (close - vwap) / atr.replace(0.0, np.nan)
        c["vwap_long_confirm"] = close > vwap
        c["vwap_short_confirm"] = close < vwap

        # Market structure breakout. shift(1) explicitly excludes current bar.
        lookback = self.config.market_structure_lookback
        prior_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=lookback).min()
        c["prior_structure_high"] = prior_high
        c["prior_structure_low"] = prior_low
        c["structure_long_confirm"] = close > prior_high
        c["structure_short_confirm"] = close < prior_low

        return c
