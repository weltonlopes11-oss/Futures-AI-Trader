from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import pandas as pd


@dataclass(frozen=True)
class OperationalTrade:
    signal_time: object
    entry_time: object
    exit_time: object
    side: str
    entry: float
    exit: float
    stop: float
    target: float
    rr: float
    outcome: str
    gross_return_pct: float
    net_return_pct: float
    r_multiple_net: float


class OperationalBacktest:
    """Convert discrete LONG/SHORT signals into executable trades.

    Causality: a signal known at candle t can only enter at candle t+1 open.
    Only one position is open at a time. Stop/target are evaluated from later
    candle high/low. If both are touched in the same candle, stop is assumed
    first (conservative, because intrabar ordering is unknown).
    """

    def __init__(self, fee_bps_per_side: float = 4.0, slippage_bps_per_side: float = 1.0):
        self.fee_bps_per_side = float(fee_bps_per_side)
        self.slippage_bps_per_side = float(slippage_bps_per_side)

    @property
    def round_trip_cost_pct(self) -> float:
        return 2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side) / 100.0

    def run(self, candles: pd.DataFrame, signals: pd.DataFrame, rr: float = 2.0) -> pd.DataFrame:
        if rr <= 0:
            raise ValueError("rr must be positive")
        required = {"timestamp", "open", "high", "low", "close", "atr"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing candle columns: {sorted(missing)}")
        if not {"timestamp", "decision"}.issubset(signals.columns):
            raise ValueError("signals require timestamp and decision")

        c = candles.sort_values("timestamp").reset_index(drop=True).copy()
        signal_map = signals.set_index("timestamp")["decision"].astype(str).str.upper().to_dict()
        trades = []
        next_available = 0

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
            slip = self.slippage_bps_per_side / 10000.0
            entry = raw_entry * (1.0 + slip if decision == "LONG" else 1.0 - slip)
            stop = entry - atr if decision == "LONG" else entry + atr
            target = entry + rr * atr if decision == "LONG" else entry - rr * atr
            exit_index = None
            raw_exit = None
            outcome = None

            for j in range(entry_index, len(c)):
                high = float(c.at[j, "high"])
                low = float(c.at[j, "low"])
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

    @staticmethod
    def metrics(trades: pd.DataFrame) -> dict:
        if trades.empty:
            return {"trades": 0, "net_return_pct": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "payoff": 0.0, "long": 0, "short": 0}
        net = trades["net_return_pct"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]
        equity = (1.0 + net / 100.0).cumprod()
        peak = equity.cummax()
        drawdown = (equity / peak - 1.0) * 100.0
        gross_profit = wins.sum()
        gross_loss = -losses.sum()
        return {
            "trades": int(len(trades)),
            "net_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
            "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
            "expectancy_pct": float(net.mean()),
            "max_drawdown_pct": float(drawdown.min()),
            "win_rate_pct": float((net > 0).mean() * 100.0),
            "payoff": float(wins.mean() / (-losses.mean())) if len(wins) and len(losses) else 0.0,
            "long": int((trades["side"] == "LONG").sum()),
            "short": int((trades["side"] == "SHORT").sum()),
        }
