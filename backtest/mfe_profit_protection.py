from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import pandas as pd

from backtest.operational_backtest import OperationalBacktest, OperationalTrade


@dataclass(frozen=True)
class MFEProfitProtectionConfig:
    activation_r: float = 0.5
    tighten_r: float = 1.0
    floor_after_activation_r: float = 0.0
    floor_after_tighten_r: float = 0.5
    structure_lookback: int = 5


class MFEProfitProtectionBacktest(OperationalBacktest):
    """Backtest with causal MFE-based profit protection.

    Frozen diagnostic rule:
    - base stop = 1 ATR and target = rr * ATR.
    - until MFE reaches +0.5R, keep the original stop.
    - once +0.5R is touched, move protected floor to entry (0R gross).
    - once +1.0R is touched, move protected floor to +0.5R gross.
    - adverse 5-bar structural close exit remains active.

    Protection is armed only after a completed bar has demonstrated the threshold.
    The newly protected stop applies from the next bar onward, avoiding same-bar
    OHLC ordering assumptions and future leakage.
    """

    def __init__(
        self,
        fee_bps_per_side: float = 4.0,
        slippage_bps_per_side: float = 1.0,
        config: MFEProfitProtectionConfig | None = None,
    ):
        super().__init__(fee_bps_per_side=fee_bps_per_side, slippage_bps_per_side=slippage_bps_per_side)
        self.config = config or MFEProfitProtectionConfig()
        if self.config.structure_lookback < 2:
            raise ValueError("structure_lookback must be >= 2")
        if not (0 <= self.config.floor_after_activation_r <= self.config.activation_r):
            raise ValueError("activation floor must lie between 0 and activation_r")
        if not (self.config.floor_after_activation_r <= self.config.floor_after_tighten_r <= self.config.tighten_r):
            raise ValueError("tighten floor must be monotonic and <= tighten_r")

    def run(self, candles: pd.DataFrame, signals: pd.DataFrame, rr: float = 2.0) -> pd.DataFrame:
        if rr <= 0:
            raise ValueError("rr must be positive")
        required = {"timestamp", "open", "high", "low", "close", "atr"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing candle columns: {sorted(missing)}")

        c = candles.sort_values("timestamp").reset_index(drop=True).copy()
        lookback = self.config.structure_lookback
        c["prior_structure_low"] = c["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        c["prior_structure_high"] = c["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        signal_map = signals.set_index("timestamp")["decision"].astype(str).str.upper().to_dict()

        trades = []
        next_available = 0
        slip = self.slippage_bps_per_side / 10000.0

        for signal_index in range(len(c) - 1):
            if signal_index < next_available:
                continue
            side = signal_map.get(c.at[signal_index, "timestamp"], "NO_TRADE")
            if side not in {"LONG", "SHORT"}:
                continue

            atr = float(c.at[signal_index, "atr"])
            if not math.isfinite(atr) or atr <= 0:
                continue

            entry_index = signal_index + 1
            raw_entry = float(c.at[entry_index, "open"])
            entry = raw_entry * (1.0 + slip if side == "LONG" else 1.0 - slip)
            initial_stop = entry - atr if side == "LONG" else entry + atr
            target = entry + rr * atr if side == "LONG" else entry - rr * atr
            active_stop = initial_stop
            protection_stage = "BASE"
            max_mfe_r = 0.0

            exit_index = None
            raw_exit = None
            outcome = None

            for j in range(entry_index, len(c)):
                high = float(c.at[j, "high"])
                low = float(c.at[j, "low"])
                close = float(c.at[j, "close"])

                stop_hit = low <= active_stop if side == "LONG" else high >= active_stop
                target_hit = high >= target if side == "LONG" else low <= target

                if stop_hit and target_hit:
                    raw_exit, outcome, exit_index = active_stop, "STOP_BOTH_TOUCHED", j
                    break
                if stop_hit:
                    raw_exit = active_stop
                    outcome = "PROTECTED_STOP" if active_stop != initial_stop else "STOP"
                    exit_index = j
                    break
                if target_hit:
                    raw_exit, outcome, exit_index = target, "TARGET", j
                    break

                prior_low = c.at[j, "prior_structure_low"]
                prior_high = c.at[j, "prior_structure_high"]
                structure_break = False
                if side == "LONG" and pd.notna(prior_low):
                    structure_break = close < float(prior_low)
                elif side == "SHORT" and pd.notna(prior_high):
                    structure_break = close > float(prior_high)
                if structure_break:
                    raw_exit, outcome, exit_index = close, "STRUCTURE_EXIT", j
                    break

                favorable = high - entry if side == "LONG" else entry - low
                bar_mfe_r = favorable / atr
                max_mfe_r = max(max_mfe_r, float(bar_mfe_r))

                # Arm protection for next bar only; this is causal under OHLC data.
                if max_mfe_r >= self.config.tighten_r:
                    protected = entry + self.config.floor_after_tighten_r * atr if side == "LONG" else entry - self.config.floor_after_tighten_r * atr
                    active_stop = max(active_stop, protected) if side == "LONG" else min(active_stop, protected)
                    protection_stage = "TIGHTENED"
                elif max_mfe_r >= self.config.activation_r:
                    protected = entry + self.config.floor_after_activation_r * atr if side == "LONG" else entry - self.config.floor_after_activation_r * atr
                    active_stop = max(active_stop, protected) if side == "LONG" else min(active_stop, protected)
                    protection_stage = "ACTIVATED"

            if exit_index is None:
                exit_index = len(c) - 1
                raw_exit = float(c.at[exit_index, "close"])
                outcome = "END_OF_DATA"

            exit_price = raw_exit * (1.0 - slip if side == "LONG" else 1.0 + slip)
            gross = ((exit_price / entry) - 1.0) * 100.0 if side == "LONG" else ((entry / exit_price) - 1.0) * 100.0
            fee_pct = 2.0 * self.fee_bps_per_side / 100.0
            net = gross - fee_pct
            risk_pct = atr / entry * 100.0

            row = asdict(
                OperationalTrade(
                    signal_time=c.at[signal_index, "timestamp"],
                    entry_time=c.at[entry_index, "timestamp"],
                    exit_time=c.at[exit_index, "timestamp"],
                    side=side,
                    entry=entry,
                    exit=exit_price,
                    stop=initial_stop,
                    target=target,
                    rr=float(rr),
                    outcome=outcome,
                    gross_return_pct=gross,
                    net_return_pct=net,
                    r_multiple_net=net / risk_pct if risk_pct else 0.0,
                )
            )
            row["protection_stage"] = protection_stage
            row["max_mfe_r_seen"] = max_mfe_r
            trades.append(row)
            next_available = exit_index + 1

        return pd.DataFrame(trades)
