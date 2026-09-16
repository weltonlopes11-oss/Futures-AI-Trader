from __future__ import annotations

import numpy as np
import pandas as pd


ENTRY_FEATURES = [
    "keltner_position",
    "keltner_width_pct",
    "adx",
    "plus_di",
    "minus_di",
    "vwap_distance_atr",
    "oi_change_pct",
    "funding_z",
    "premium_z",
    "cvd_fast",
    "cvd_slow",
    "distance_to_swing_high_atr",
    "distance_to_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "structure_regime",
    "swing_high_class",
    "swing_low_class",
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "double_top",
    "double_bottom",
]


class TradeQualityAnalytics:
    """Causal post-trade diagnostics for already-generated backtest trades.

    MFE/MAE are measured only between the actual entry and exit timestamps.
    They are normalized by the trade's initial 1R risk distance (|entry-stop|).
    Entry-state features are joined at signal_time, i.e. the last completed bar
    known before the next-open execution.
    """

    @staticmethod
    def _side_excursions(side: str, entry: float, highs: pd.Series, lows: pd.Series):
        if side == "LONG":
            favorable = highs.astype(float) - entry
            adverse = entry - lows.astype(float)
        elif side == "SHORT":
            favorable = entry - lows.astype(float)
            adverse = highs.astype(float) - entry
        else:
            raise ValueError(f"unknown side: {side}")
        return favorable, adverse

    def enrich_trades(self, candles: pd.DataFrame, context: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades.copy()
        required_trade = {"signal_time", "entry_time", "exit_time", "side", "entry", "stop", "net_return_pct"}
        missing_trade = required_trade - set(trades.columns)
        if missing_trade:
            raise ValueError(f"trades missing columns: {sorted(missing_trade)}")
        required_candle = {"timestamp", "high", "low", "close"}
        missing_candle = required_candle - set(candles.columns)
        if missing_candle:
            raise ValueError(f"candles missing columns: {sorted(missing_candle)}")
        if "timestamp" not in context.columns:
            raise ValueError("context missing timestamp")

        c = candles.copy().sort_values("timestamp").reset_index(drop=True)
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
        ctx = context.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
        out_rows = []

        feature_cols = [col for col in ENTRY_FEATURES if col in ctx.columns]
        ctx_indexed = ctx.set_index("timestamp")

        for _, trade in trades.iterrows():
            signal_time = pd.to_datetime(trade["signal_time"], utc=True)
            entry_time = pd.to_datetime(trade["entry_time"], utc=True)
            exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            path = c[(c["timestamp"] >= entry_time) & (c["timestamp"] <= exit_time)].copy()
            if path.empty:
                raise ValueError(f"no candle path for trade at {entry_time}")

            entry = float(trade["entry"])
            stop = float(trade["stop"])
            risk = abs(entry - stop)
            if not np.isfinite(risk) or risk <= 0:
                raise ValueError("trade risk distance must be positive")

            favorable, adverse = self._side_excursions(str(trade["side"]).upper(), entry, path["high"], path["low"])
            mfe_idx = int(favorable.to_numpy().argmax())
            mae_idx = int(adverse.to_numpy().argmax())
            mfe = max(0.0, float(favorable.iloc[mfe_idx]))
            mae = max(0.0, float(adverse.iloc[mae_idx]))

            row = trade.to_dict()
            row.update(
                {
                    "mfe_r": mfe / risk,
                    "mae_r": mae / risk,
                    "mfe_pct": mfe / entry * 100.0,
                    "mae_pct": mae / entry * 100.0,
                    "bars_to_mfe": mfe_idx,
                    "bars_to_mae": mae_idx,
                    "trade_duration_bars": max(0, len(path) - 1),
                    "reached_0_5r": bool(mfe / risk >= 0.5),
                    "reached_1r": bool(mfe / risk >= 1.0),
                    "reached_2r": bool(mfe / risk >= 2.0),
                    "winner": bool(float(trade["net_return_pct"]) > 0.0),
                }
            )

            if signal_time in ctx_indexed.index:
                snapshot = ctx_indexed.loc[signal_time]
                if isinstance(snapshot, pd.DataFrame):
                    snapshot = snapshot.iloc[-1]
                for col in feature_cols:
                    row[f"entry_{col}"] = snapshot[col]

                if "adx" in ctx.columns:
                    prior = ctx[ctx["timestamp"] < signal_time].tail(1)
                    row["entry_adx_delta"] = (
                        float(snapshot["adx"]) - float(prior.iloc[0]["adx"])
                        if len(prior) and pd.notna(snapshot["adx"]) and pd.notna(prior.iloc[0]["adx"])
                        else np.nan
                    )
                if "cvd_fast" in ctx.columns and "cvd_slow" in ctx.columns:
                    row["entry_cvd_spread"] = float(snapshot["cvd_fast"]) - float(snapshot["cvd_slow"])

            out_rows.append(row)

        return pd.DataFrame(out_rows)

    @staticmethod
    def winner_loser_summary(enriched: pd.DataFrame) -> pd.DataFrame:
        if enriched.empty:
            return pd.DataFrame()
        numeric = [
            "net_return_pct", "mfe_r", "mae_r", "bars_to_mfe", "bars_to_mae", "trade_duration_bars",
            "entry_keltner_position", "entry_adx", "entry_adx_delta", "entry_vwap_distance_atr",
            "entry_oi_change_pct", "entry_funding_z", "entry_premium_z",
            "entry_distance_to_swing_high_atr", "entry_distance_to_swing_low_atr",
            "entry_bars_since_swing_high", "entry_bars_since_swing_low",
        ]
        numeric = [c for c in numeric if c in enriched.columns]
        rows = []
        for winner, grp in enriched.groupby("winner", dropna=False):
            row = {"group": "winner" if winner else "loser", "trades": int(len(grp))}
            for col in numeric:
                values = pd.to_numeric(grp[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                row[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
