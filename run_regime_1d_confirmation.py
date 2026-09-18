from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import requests
import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_regime_4h_ema50_backtest import attach_4h_regime
from run_regime_discovery_2025_validate_2026 import add_regime_features

OFFICIAL_REGIMES = {"LONG|LOW|CHOP", "SHORT|NORMAL|MIXED"}


class MonthlyBinanceLoader(BinanceDataVisionLoader):
    BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

    def fetch_window(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        frames = []
        year, month = start.year, start.month
        end_year, end_month = end.year, end.month
        while (year, month) <= (end_year, end_month):
            stamp = f"{year:04d}-{month:02d}"
            url = f"{self.BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"
            response = self.session.get(url, timeout=60)
            if response.status_code == 404:
                year, month = (year + 1, 1) if month == 12 else (year, month + 1)
                continue
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as archive:
                frame = pd.read_csv(archive.open(archive.namelist()[0]), header=None, names=self.COLUMNS)
            frame["timestamp"] = self._parse_epoch(frame["timestamp"])
            frame["close_time"] = self._parse_epoch(frame["close_time"])
            for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
            frame["close_time"] = frame["close_time"].dt.tz_localize(None)
            frames.append(frame.dropna(subset=["timestamp", "open", "high", "low", "close"]))
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        if not frames:
            raise RuntimeError(f"No monthly Binance data for {symbol} {interval}")
        result = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
        end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)
        result = result[(result["timestamp"] >= start_naive) & (result["timestamp"] < end_naive)].reset_index(drop=True)
        if result.empty:
            raise RuntimeError(f"Monthly Binance data returned no {symbol} {interval} candles")
        return result
EMA_1D_PERIOD = 50
INITIAL_EQUITY_BRL = 2000.0
RISK_FRACTION = 0.10
MAX_LEVERAGE = 15.0


def build_1d_regime(d1: pd.DataFrame) -> pd.DataFrame:
    out = d1.copy().sort_values("timestamp").reset_index(drop=True)
    out["ema50_1d"] = out["close"].ewm(
        span=EMA_1D_PERIOD, adjust=False, min_periods=EMA_1D_PERIOD
    ).mean()
    out["ema50_1d_prev"] = out["ema50_1d"].shift(1)
    out["ema50_1d_slope"] = out["ema50_1d"] - out["ema50_1d_prev"]

    bullish = (out["close"] > out["ema50_1d"]) & (out["ema50_1d_slope"] > 0)
    bearish = (out["close"] < out["ema50_1d"]) & (out["ema50_1d_slope"] < 0)
    out["regime_1d"] = np.select(
        [bullish, bearish], ["BULLISH", "BEARISH"], default="NEUTRAL"
    )
    out["regime_1d_close"] = out["close"].astype(float)
    # Binance daily timestamps are bar-open times. Use only a fully closed 1D candle.
    out["regime_1d_available_time"] = pd.to_datetime(out["timestamp"]) + pd.Timedelta(days=1)
    return out[
        [
            "regime_1d_available_time",
            "regime_1d",
            "regime_1d_close",
            "ema50_1d",
            "ema50_1d_slope",
        ]
    ].dropna(subset=["ema50_1d", "ema50_1d_slope"])


def attach_1d_regime(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    one = h1.copy().sort_values("timestamp").reset_index(drop=True)
    if "signal_available_time" not in one.columns:
        one["signal_available_time"] = pd.to_datetime(one["timestamp"]) + pd.Timedelta(hours=1)
    daily = build_1d_regime(d1).sort_values("regime_1d_available_time")
    return pd.merge_asof(
        one.sort_values("signal_available_time"),
        daily,
        left_on="signal_available_time",
        right_on="regime_1d_available_time",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("timestamp").reset_index(drop=True)


def official_risk_path(trades: pd.DataFrame) -> dict:
    equity = INITIAL_EQUITY_BRL
    peak = equity
    max_dd = 0.0
    rows = []
    for _, trade in trades.sort_values("entry_time").iterrows():
        entry = float(trade["entry_price"])
        atr = float(trade["entry_atr"])
        stop_distance = (2.0 * atr) / entry
        leverage = min(MAX_LEVERAGE, RISK_FRACTION / stop_distance)
        notional = equity * leverage
        pnl = notional * float(trade["net_return_pct"]) / 100.0
        before = equity
        equity += pnl
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        max_dd = min(max_dd, dd)
        rows.append(
            {
                "entry_time": str(trade["entry_time"]),
                "side": trade["side"],
                "regime_key": trade["regime_key"],
                "stop_distance_pct": stop_distance * 100.0,
                "leverage": leverage,
                "equity_before_brl": before,
                "pnl_brl": pnl,
                "equity_after_brl": equity,
                "drawdown_pct": dd * 100.0,
            }
        )
    return {
        "initial_equity_brl": INITIAL_EQUITY_BRL,
        "final_equity_brl": equity,
        "return_pct": (equity / INITIAL_EQUITY_BRL - 1.0) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "trades": len(rows),
        "path": rows,
    }


def run_window(
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    start: datetime,
    end: datetime,
    cfg: IchimokuKeltner1HConfig,
    require_1d_alignment: bool,
) -> tuple[pd.DataFrame, dict]:
    enriched = add_regime_features(enrich_indicators(h1, cfg))
    enriched = attach_4h_regime(enriched, h4)
    enriched = attach_1d_regime(enriched, d1)

    start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
    end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)

    trades: list[dict] = []
    position = None
    pending = None
    counts = {
        "accepted": 0,
        "blocked_4h": 0,
        "blocked_official_regime": 0,
        "blocked_1d": 0,
    }
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0

    for i, row in enriched.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        in_eval = start_naive <= ts < end_naive

        if position is None and pending is not None and in_eval:
            prior_atr = enriched.iloc[i - 1]["keltner_atr"] if i > 0 else float("nan")
            if pd.isna(prior_atr):
                pending = None
            else:
                entry = float(row["open"])
                atr = float(prior_atr)
                side = pending["side"]
                position = {
                    **pending,
                    "entry_time": ts,
                    "entry_price": entry,
                    "entry_atr": atr,
                    "stop_price": entry - 2.0 * atr if side == "LONG" else entry + 2.0 * atr,
                    "entry_index": i,
                }
                pending = None
        elif not in_eval:
            pending = None

        if position is not None and in_eval:
            side = position["side"]
            stop = float(position["stop_price"])
            o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
            if side == "LONG":
                target = row["executable_keltner_upper"]
                if l <= stop:
                    exit_price, reason = min(o, stop), "ATR2_PROTECTIVE_STOP"
                elif pd.notna(target) and h >= float(target):
                    exit_price, reason = max(o, float(target)), "KELTNER_UPPER_TOUCH"
                else:
                    exit_price = reason = None
            else:
                target = row["executable_keltner_lower"]
                if h >= stop:
                    exit_price, reason = max(o, stop), "ATR2_PROTECTIVE_STOP"
                elif pd.notna(target) and l <= float(target):
                    exit_price, reason = min(o, float(target)), "KELTNER_LOWER_TOUCH"
                else:
                    exit_price = reason = None

            if exit_price is not None:
                gross = (
                    exit_price / position["entry_price"] - 1.0
                    if side == "LONG"
                    else position["entry_price"] / exit_price - 1.0
                ) * 100.0
                trades.append(
                    {
                        **position,
                        "exit_time": ts,
                        "exit_price": exit_price,
                        "exit_reason": reason,
                        "bars_held": i - position["entry_index"] + 1,
                        "gross_return_pct": gross,
                        "cost_pct": cost,
                        "net_return_pct": gross - cost,
                    }
                )
                position = None

        if in_eval and position is None and pending is None and i + 1 < len(enriched):
            if pd.Timestamp(enriched.iloc[i + 1]["timestamp"]) >= end_naive:
                continue

            if bool(row["long_signal"]):
                side, required = "LONG", "BULLISH"
            elif bool(row["short_signal"]):
                side, required = "SHORT", "BEARISH"
            else:
                continue

            regime_4h = str(row.get("regime_4h", "NEUTRAL"))
            if regime_4h != required:
                counts["blocked_4h"] += 1
                continue

            vol = str(row.get("vol_regime", "UNKNOWN"))
            eff = str(row.get("efficiency_regime", "UNKNOWN"))
            key = f"{side}|{vol}|{eff}"
            if key not in OFFICIAL_REGIMES:
                counts["blocked_official_regime"] += 1
                continue

            regime_1d = str(row.get("regime_1d", "NEUTRAL"))
            if require_1d_alignment and regime_1d != required:
                counts["blocked_1d"] += 1
                continue

            counts["accepted"] += 1
            pending = {
                "side": side,
                "signal_time": row["timestamp"],
                "regime_key": key,
                "vol_regime": vol,
                "efficiency_regime": eff,
                "atr_price": float(row["atr_price"]),
                "er20": float(row["er20"]),
                "regime_4h": regime_4h,
                "regime_1d": regime_1d,
                "ema50_4h": float(row["ema50_4h"]),
                "ema50_1d": float(row["ema50_1d"]),
                "ema50_1d_slope": float(row["ema50_1d_slope"]),
            }

    if position is not None:
        eligible = enriched[(enriched["timestamp"] >= start_naive) & (enriched["timestamp"] < end_naive)]
        last = eligible.iloc[-1]
        exit_price = float(last["close"])
        gross = (
            exit_price / position["entry_price"] - 1.0
            if position["side"] == "LONG"
            else position["entry_price"] / exit_price - 1.0
        ) * 100.0
        trades.append(
            {
                **position,
                "exit_time": last["timestamp"],
                "exit_price": exit_price,
                "exit_reason": "EVAL_END_MTM",
                "bars_held": int(eligible.index[-1]) - position["entry_index"] + 1,
                "gross_return_pct": gross,
                "cost_pct": cost,
                "net_return_pct": gross - cost,
            }
        )

    return pd.DataFrame(trades), counts


def summarize(trades: pd.DataFrame) -> dict:
    return {
        "metrics": metrics(trades),
        "official_risk_path": official_risk_path(trades),
    }


def main() -> None:
    cfg = IchimokuKeltner1HConfig(fee_bps_per_side=4.0, slippage_bps_per_side=1.0)
    periods = {
        "2024": (datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc)),
        "2025": (datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)),
        "2026_JAN_AUG": (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)),
    }
    loader = MonthlyBinanceLoader()
    warmup = periods["2024"][0] - timedelta(days=150)
    end = periods["2026_JAN_AUG"][1]
    h1 = loader.fetch_window("ETHUSDT", "1h", warmup, end)
    h4 = loader.fetch_window("ETHUSDT", "4h", warmup - timedelta(days=20), end)
    d1 = loader.fetch_window("ETHUSDT", "1d", warmup - timedelta(days=120), end)

    out = Path("artifacts/regime-1d-confirmation")
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "research_only",
        "official_project_modified": False,
        "hypothesis": "Require latest fully completed 1D EMA50 direction+slope to align with side, in addition to current official 4H + adaptive-volatility + ER20 regime gate.",
        "daily_rule": "BULLISH close>EMA50 and EMA50 slope>0; BEARISH close<EMA50 and EMA50 slope<0; causal availability at daily close.",
        "periods": {},
    }

    for name, (start, finish) in periods.items():
        baseline, baseline_counts = run_window(h1, h4, d1, start, finish, cfg, False)
        aligned, aligned_counts = run_window(h1, h4, d1, start, finish, cfg, True)
        result["periods"][name] = {
            "current_official_gate": {**summarize(baseline), "counts": baseline_counts},
            "plus_1d_alignment": {**summarize(aligned), "counts": aligned_counts},
        }
        baseline.to_csv(out / f"{name.lower()}_baseline.csv", index=False)
        aligned.to_csv(out / f"{name.lower()}_plus_1d.csv", index=False)

    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
