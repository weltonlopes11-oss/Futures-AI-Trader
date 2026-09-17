from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeveragedCapitalConfig:
    initial_capital: float = 500.0
    leverage: float = 10.0
    margin_fraction: float = 1.0
    maintenance_margin_rate: float = 0.005

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if not 0 < self.margin_fraction <= 1:
            raise ValueError("margin_fraction must be in (0, 1]")
        if not 0 <= self.maintenance_margin_rate < 1 / self.leverage:
            raise ValueError("maintenance_margin_rate must be below initial margin rate")


class LeveragedCapitalSimulator:
    """Compound already-net trade returns onto margin capital.

    `net_return_pct` is the strategy return on notional after the backtest's
    fees and slippage. Margin return is therefore multiplied by leverage.

    Liquidation risk is diagnostic only: for isolated margin, the approximate
    adverse price threshold is initial-margin-rate minus maintenance-margin-rate.
    Exact exchange liquidation depends on mark price, tier, fees and wallet mode.
    """

    def __init__(self, config: LeveragedCapitalConfig | None = None):
        self.config = config or LeveragedCapitalConfig()

    @property
    def approximate_liquidation_adverse_pct(self) -> float:
        return (1.0 / self.config.leverage - self.config.maintenance_margin_rate) * 100.0

    @staticmethod
    def _adverse_excursion_pct(side: str, entry: float, highs: pd.Series, lows: pd.Series) -> float:
        side = side.upper()
        if side == "LONG":
            adverse = (entry - lows.astype(float)) / entry * 100.0
        elif side == "SHORT":
            adverse = (highs.astype(float) - entry) / entry * 100.0
        else:
            raise ValueError(f"unknown side: {side}")
        return max(0.0, float(adverse.max()))

    def simulate(self, candles: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        required_trade = {"entry_time", "exit_time", "side", "entry", "net_return_pct", "outcome"}
        missing = required_trade - set(trades.columns)
        if missing:
            raise ValueError(f"trades missing columns: {sorted(missing)}")
        required_candle = {"timestamp", "high", "low"}
        missing_candle = required_candle - set(candles.columns)
        if missing_candle:
            raise ValueError(f"candles missing columns: {sorted(missing_candle)}")

        c = candles.copy().sort_values("timestamp").reset_index(drop=True)
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
        capital = float(self.config.initial_capital)
        peak = capital
        liq_threshold = self.approximate_liquidation_adverse_pct
        rows: list[dict] = []

        for i, trade in trades.reset_index(drop=True).iterrows():
            entry_time = pd.to_datetime(trade["entry_time"], utc=True)
            exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            path = c[(c["timestamp"] >= entry_time) & (c["timestamp"] <= exit_time)]
            if path.empty:
                raise ValueError(f"no candle path for trade {i}")

            capital_before = capital
            margin_used = capital_before * self.config.margin_fraction
            notional = margin_used * self.config.leverage
            net_notional_return = float(trade["net_return_pct"]) / 100.0
            pnl = notional * net_notional_return

            max_adverse_pct = self._adverse_excursion_pct(
                str(trade["side"]), float(trade["entry"]), path["high"], path["low"]
            )
            liquidation_risk = max_adverse_pct >= liq_threshold
            liquidation_buffer_pct = liq_threshold - max_adverse_pct

            # If the modeled path crosses the approximate liquidation threshold,
            # cap the allocated isolated margin at zero. Otherwise apply trade P&L.
            if liquidation_risk:
                capital = capital_before - margin_used
                pnl = -margin_used
            else:
                capital = capital_before + pnl

            capital = max(0.0, capital)
            peak = max(peak, capital)
            drawdown_pct = (capital / peak - 1.0) * 100.0 if peak > 0 else -100.0
            margin_return_pct = (pnl / margin_used * 100.0) if margin_used > 0 else 0.0

            row = trade.to_dict()
            row.update(
                {
                    "trade_number": int(i + 1),
                    "capital_before": capital_before,
                    "margin_used": margin_used,
                    "notional": notional,
                    "leveraged_pnl": pnl,
                    "margin_return_pct": margin_return_pct,
                    "capital_after": capital,
                    "equity_drawdown_pct": drawdown_pct,
                    "max_adverse_price_pct": max_adverse_pct,
                    "approx_liquidation_adverse_pct": liq_threshold,
                    "liquidation_buffer_pct": liquidation_buffer_pct,
                    "approx_liquidation_hit": bool(liquidation_risk),
                }
            )
            rows.append(row)

            if capital <= 0:
                break

        return pd.DataFrame(rows)

    def summary(self, simulated: pd.DataFrame) -> dict:
        if simulated.empty:
            return {
                "initial_capital": self.config.initial_capital,
                "final_capital": self.config.initial_capital,
                "net_profit": 0.0,
                "return_on_initial_pct": 0.0,
                "max_equity_drawdown_pct": 0.0,
                "liquidation_hits": 0,
            }
        final = float(simulated.iloc[-1]["capital_after"])
        initial = float(self.config.initial_capital)
        return {
            "initial_capital": initial,
            "final_capital": final,
            "net_profit": final - initial,
            "return_on_initial_pct": (final / initial - 1.0) * 100.0,
            "max_equity_drawdown_pct": float(simulated["equity_drawdown_pct"].min()),
            "liquidation_hits": int(simulated["approx_liquidation_hit"].sum()),
            "minimum_liquidation_buffer_pct": float(simulated["liquidation_buffer_pct"].min()),
            "maximum_adverse_price_pct": float(simulated["max_adverse_price_pct"].max()),
            "trades": int(len(simulated)),
            "leverage": self.config.leverage,
            "margin_fraction": self.config.margin_fraction,
            "maintenance_margin_rate": self.config.maintenance_margin_rate,
            "approx_liquidation_adverse_pct": self.approximate_liquidation_adverse_pct,
        }
