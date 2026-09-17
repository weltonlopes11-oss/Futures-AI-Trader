from __future__ import annotations

from dataclasses import asdict
import math

import pandas as pd

from backtest.operational_backtest import OperationalBacktest, OperationalTrade


class ManagedExitBacktest(OperationalBacktest):
    """Causal trade management layered on the frozen stop/target semantics.

    Rules are predeclared and non-fitted:
    - initial stop = 1 ATR and target = fixed R multiple;
    - adverse 5-bar structure break at candle close remains active;
    - after a completed candle CLOSES at +1R or better, break-even becomes
      active only from the NEXT candle;
    - once break-even is active, a close through the Keltner middle exits the
      position if stop/target/structure have not already exited it.

    Stop/target (including an already-active break-even stop) are checked first
    intrabar. Close-based exits are evaluated afterwards, avoiding look-ahead.
    """

    def __init__(
        self,
        fee_bps_per_side: float = 4.0,
        slippage_bps_per_side: float = 1.0,
        structure_lookback: int = 5,
    ):
        super().__init__(fee_bps_per_side=fee_bps_per_side, slippage_bps_per_side=slippage_bps_per_side)
        if structure_lookback < 2:
            raise ValueError("structure_lookback must be >= 2")
        self.structure_lookback = int(structure_lookback)

    def run(self, candles: pd.DataFrame, signals: pd.DataFrame, rr: float = 2.0) -> pd.DataFrame:
        if rr <= 0:
            raise ValueError("rr must be positive")
        required = {"timestamp", "open", "high", "low", "close", "atr", "keltner_middle"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing candle columns: {sorted(missing)}")
        if not {"timestamp", "decision"}.issubset(signals.columns):
            raise ValueError("signals require timestamp and decision")

        c = candles.sort_values("timestamp").reset_index(drop=True).copy()
        c["prior_structure_low_5"] = c["low"].shift(1).rolling(
            self.structure_lookback, min_periods=self.structure_lookback
        ).min()
        c["prior_structure_high_5"] = c["high"].shift(1).rolling(
            self.structure_lookback, min_periods=self.structure_lookback
        ).max()

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
            initial_stop = entry - atr if decision == "LONG" else entry + atr
            target = entry + rr * atr if decision == "LONG" else entry - rr * atr
            one_r_level = entry + atr if decision == "LONG" else entry - atr

            breakeven_active = False
            exit_index = None
            raw_exit = None
            outcome = None

            for j in range(entry_index, len(c)):
                high = float(c.at[j, "high"])
                low = float(c.at[j, "low"])
                close = float(c.at[j, "close"])

                active_stop = entry if breakeven_active else initial_stop
                stop_hit = low <= active_stop if decision == "LONG" else high >= active_stop
                target_hit = high >= target if decision == "LONG" else low <= target

                if stop_hit and target_hit:
                    raw_exit = active_stop
                    outcome = "BREAKEVEN_BOTH_TOUCHED" if breakeven_active else "STOP_BOTH_TOUCHED"
                    exit_index = j
                    break
                if stop_hit:
                    raw_exit = active_stop
                    outcome = "BREAKEVEN" if breakeven_active else "STOP"
                    exit_index = j
                    break
                if target_hit:
                    raw_exit = target
                    outcome = "TARGET"
                    exit_index = j
                    break

                prior_low = c.at[j, "prior_structure_low_5"]
                prior_high = c.at[j, "prior_structure_high_5"]
                structure_break = False
                if decision == "LONG" and pd.notna(prior_low):
                    structure_break = close < float(prior_low)
                elif decision == "SHORT" and pd.notna(prior_high):
                    structure_break = close > float(prior_high)
                if structure_break:
                    raw_exit, outcome, exit_index = close, "STRUCTURE_EXIT", j
                    break

                middle = c.at[j, "keltner_middle"]
                if breakeven_active and pd.notna(middle):
                    keltner_lost = (
                        (decision == "LONG" and close < float(middle))
                        or (decision == "SHORT" and close > float(middle))
                    )
                    if keltner_lost:
                        raw_exit, outcome, exit_index = close, "KELTNER_MIDDLE_EXIT", j
                        break

                # Activation occurs only after this candle is fully known and is
                # therefore effective from the next iteration/candle.
                if not breakeven_active:
                    if decision == "LONG" and close >= one_r_level:
                        breakeven_active = True
                    elif decision == "SHORT" and close <= one_r_level:
                        breakeven_active = True

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
                stop=initial_stop,
                target=target,
                rr=float(rr),
                outcome=outcome,
                gross_return_pct=gross,
                net_return_pct=net,
                r_multiple_net=net / risk_pct if risk_pct else 0.0,
            )))
            next_available = exit_index + 1

        return pd.DataFrame(trades)
