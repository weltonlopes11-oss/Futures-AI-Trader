from __future__ import annotations

from dataclasses import asdict
import math

import pandas as pd

from backtest.operational_backtest import OperationalBacktest, OperationalTrade


class ExitVariantsBacktest(OperationalBacktest):
    """Compare causal exit variants while keeping entry/stop/target unchanged.

    Modes:
    - five_bar: control, adverse close break of prior 5 bars.
    - three_bar: adverse close break of prior 3 bars.
    - keltner_middle: adverse close cross of the causal Keltner middle line.
    - adaptive_05r_three_bar: use 5-bar structure until price reaches +0.5R,
      then tighten to a 3-bar adverse structural close break.

    Stop/target checks always occur before close-based exits on a candle. If stop
    and target are both touched in the same candle, STOP wins, matching the
    frozen research protocol.
    """

    VALID_MODES = {"five_bar", "three_bar", "keltner_middle", "adaptive_05r_three_bar"}

    def __init__(self, fee_bps_per_side: float = 4.0, slippage_bps_per_side: float = 1.0):
        super().__init__(fee_bps_per_side=fee_bps_per_side, slippage_bps_per_side=slippage_bps_per_side)

    @staticmethod
    def _add_structure(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
        c = frame.copy()
        c[f"prior_low_{lookback}"] = c["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        c[f"prior_high_{lookback}"] = c["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        return c

    def run(self, candles: pd.DataFrame, signals: pd.DataFrame, rr: float = 2.0, mode: str = "five_bar") -> pd.DataFrame:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown exit mode: {mode}")
        if rr <= 0:
            raise ValueError("rr must be positive")

        required = {"timestamp", "open", "high", "low", "close", "atr"}
        if mode == "keltner_middle":
            required.add("keltner_middle")
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing candle columns: {sorted(missing)}")
        if not {"timestamp", "decision"}.issubset(signals.columns):
            raise ValueError("signals require timestamp and decision")

        c = candles.sort_values("timestamp").reset_index(drop=True).copy()
        c = self._add_structure(c, 3)
        c = self._add_structure(c, 5)
        signal_map = signals.set_index("timestamp")["decision"].astype(str).str.upper().to_dict()

        trades: list[dict] = []
        next_available = 0
        slip = self.slippage_bps_per_side / 10000.0

        for signal_index in range(len(c) - 1):
            if signal_index < next_available:
                continue
            decision = signal_map.get(c.at[signal_index, "timestamp"], "NO_TRADE")
            if decision not in {"LONG", "SHORT"}:
                continue

            atr = float(c.at[signal_index, "atr"])
            if not math.isfinite(atr) or atr <= 0:
                continue

            entry_index = signal_index + 1
            raw_entry = float(c.at[entry_index, "open"])
            entry = raw_entry * (1.0 + slip if decision == "LONG" else 1.0 - slip)
            stop = entry - atr if decision == "LONG" else entry + atr
            target = entry + rr * atr if decision == "LONG" else entry - rr * atr

            exit_index = None
            raw_exit = None
            outcome = None
            armed_05r = False
            max_favorable_r = 0.0

            for j in range(entry_index, len(c)):
                high = float(c.at[j, "high"])
                low = float(c.at[j, "low"])
                close = float(c.at[j, "close"])

                favorable_r = (high - entry) / atr if decision == "LONG" else (entry - low) / atr
                max_favorable_r = max(max_favorable_r, favorable_r)
                if favorable_r >= 0.5:
                    armed_05r = True

                stop_hit = low <= stop if decision == "LONG" else high >= stop
                target_hit = high >= target if decision == "LONG" else low <= target
                if stop_hit and target_hit:
                    raw_exit, outcome, exit_index = stop, "STOP_BOTH_TOUCHED", j
                    break
                if stop_hit:
                    raw_exit, outcome, exit_index = stop, "STOP", j
                    break
                if target_hit:
                    raw_exit, outcome, exit_index = target, "TARGET", j
                    break

                structure_break = False
                exit_label = None
                if mode in {"five_bar", "three_bar", "adaptive_05r_three_bar"}:
                    lookback = 5
                    if mode == "three_bar" or (mode == "adaptive_05r_three_bar" and armed_05r):
                        lookback = 3
                    prior_low = c.at[j, f"prior_low_{lookback}"]
                    prior_high = c.at[j, f"prior_high_{lookback}"]
                    if decision == "LONG" and pd.notna(prior_low):
                        structure_break = close < float(prior_low)
                    elif decision == "SHORT" and pd.notna(prior_high):
                        structure_break = close > float(prior_high)
                    if structure_break:
                        exit_label = f"STRUCTURE_EXIT_{lookback}"
                elif mode == "keltner_middle":
                    middle = c.at[j, "keltner_middle"]
                    if pd.notna(middle):
                        structure_break = close < float(middle) if decision == "LONG" else close > float(middle)
                    if structure_break:
                        exit_label = "KELTNER_MIDDLE_EXIT"

                if structure_break:
                    raw_exit, outcome, exit_index = close, exit_label, j
                    break

            if exit_index is None:
                exit_index = len(c) - 1
                raw_exit = float(c.at[exit_index, "close"])
                outcome = "END_OF_DATA"

            exit_price = raw_exit * (1.0 - slip if decision == "LONG" else 1.0 + slip)
            gross = ((exit_price / entry) - 1.0) * 100.0
            if decision == "SHORT":
                gross = ((entry / exit_price) - 1.0) * 100.0
            fee_pct = 2.0 * self.fee_bps_per_side / 100.0
            net = gross - fee_pct
            risk_pct = atr / entry * 100.0
            realized_r = net / risk_pct if risk_pct else 0.0

            row = asdict(
                OperationalTrade(
                    signal_time=c.at[signal_index, "timestamp"],
                    entry_time=c.at[entry_index, "timestamp"],
                    exit_time=c.at[exit_index, "timestamp"],
                    side=decision,
                    entry=entry,
                    exit=exit_price,
                    stop=stop,
                    target=target,
                    rr=float(rr),
                    outcome=outcome,
                    gross_return_pct=gross,
                    net_return_pct=net,
                    r_multiple_net=realized_r,
                )
            )
            row["max_favorable_r_raw"] = max_favorable_r
            row["mfe_giveback_r"] = max_favorable_r - realized_r
            row["adaptive_armed_05r"] = armed_05r
            trades.append(row)
            next_available = exit_index + 1

        return pd.DataFrame(trades)
