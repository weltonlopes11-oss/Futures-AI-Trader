from __future__ import annotations

from dataclasses import asdict
import math

import pandas as pd

from backtest.operational_backtest import OperationalBacktest, OperationalTrade


class PivotStructureExitBacktest(OperationalBacktest):
    """Exit on invalidation of the latest confirmed causal swing.

    LONG exits at candle close if close < latest confirmed swing low.
    SHORT exits at candle close if close > latest confirmed swing high.
    Stop/target retain intrabar priority. Swing columns must already be produced
    causally by PriceStructurePatterns.
    """

    def run(self, candles: pd.DataFrame, signals: pd.DataFrame, rr: float = 2.0) -> pd.DataFrame:
        if rr <= 0:
            raise ValueError("rr must be positive")
        required = {"timestamp", "open", "high", "low", "close", "atr", "last_swing_high", "last_swing_low"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing candle columns: {sorted(missing)}")
        if not {"timestamp", "decision"}.issubset(signals.columns):
            raise ValueError("signals require timestamp and decision")

        c = candles.sort_values("timestamp").reset_index(drop=True).copy()
        signal_map = signals.set_index("timestamp")["decision"].astype(str).str.upper().to_dict()
        trades = []
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

            for j in range(entry_index, len(c)):
                high = float(c.at[j, "high"])
                low = float(c.at[j, "low"])
                close = float(c.at[j, "close"])
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

                if decision == "LONG":
                    level = c.at[j, "last_swing_low"]
                    invalid = pd.notna(level) and close < float(level)
                else:
                    level = c.at[j, "last_swing_high"]
                    invalid = pd.notna(level) and close > float(level)

                if invalid:
                    raw_exit, outcome, exit_index = close, "PIVOT_STRUCTURE_EXIT", j
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
            trades.append(asdict(OperationalTrade(
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
                r_multiple_net=net / risk_pct if risk_pct else 0.0,
            )))
            next_available = exit_index + 1

        return pd.DataFrame(trades)
