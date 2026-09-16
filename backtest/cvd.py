from __future__ import annotations

import pandas as pd


class CumulativeVolumeDelta:
    """Build a causal CVD feature from Binance USD-M kline taker volume.

    Binance futures klines expose total base-asset volume and taker-buy base
    volume. Because total volume = taker buys + taker sells, signed delta is:

        delta = taker_buy_base - taker_sell_base
              = 2 * taker_buy_base - volume

    CVD is the cumulative sum of that signed delta. Direction is measured with
    the same 12/26 EMA convention already used by the price trend model, which
    avoids introducing a newly tuned lookback solely for CVD.
    """

    def __init__(self, fast: int = 12, slow: int = 26):
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast < slow")
        self.fast = int(fast)
        self.slow = int(slow)

    def enrich(self, candles: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "volume", "taker_buy_base"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing CVD columns: {sorted(missing)}")

        frame = candles.copy().sort_values("timestamp").reset_index(drop=True)
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame["taker_buy_base"] = pd.to_numeric(frame["taker_buy_base"], errors="coerce")

        if frame[["volume", "taker_buy_base"]].isna().any().any():
            raise ValueError("CVD input contains non-numeric volume values")
        if (frame["volume"] < 0).any() or (frame["taker_buy_base"] < 0).any():
            raise ValueError("CVD volumes must be non-negative")
        if (frame["taker_buy_base"] > frame["volume"] + 1e-12).any():
            raise ValueError("taker_buy_base cannot exceed total volume")

        frame["taker_sell_base"] = frame["volume"] - frame["taker_buy_base"]
        frame["cvd_delta"] = frame["taker_buy_base"] - frame["taker_sell_base"]
        frame["cvd"] = frame["cvd_delta"].cumsum()

        fast_ema = frame["cvd"].ewm(span=self.fast, adjust=False).mean()
        slow_ema = frame["cvd"].ewm(span=self.slow, adjust=False).mean()
        frame["cvd_signal"] = pd.Series(pd.NA, index=frame.index, dtype="object")
        frame.loc[fast_ema > slow_ema, "cvd_signal"] = "LONG"
        frame.loc[fast_ema < slow_ema, "cvd_signal"] = "SHORT"
        return frame
