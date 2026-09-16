from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import pandas as pd


@dataclass(slots=True)
class OIProfitObservation:
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    decision: str
    regime: str
    market_direction: str
    oi_change_pct: float
    oi_bucket: str
    entry_price: float
    horizon_minutes: int
    future_return_pct: float
    directional_return_pct: float
    net_return_pct: float
    mfe_pct: float
    mae_pct: float


class OIProfitAnalyzer:
    DEFAULT_HORIZONS = (5, 15, 30, 60)

    def __init__(self, records: Iterable, candles: pd.DataFrame, horizons=DEFAULT_HORIZONS,
                 fee_bps_per_side: float = 0.0, slippage_bps_per_side: float = 0.0):
        self.records = list(records)
        self.candles = self._prepare_candles(candles)
        self.horizons = tuple(sorted(set(int(x) for x in horizons if int(x) > 0)))
        self.fee_bps_per_side = max(float(fee_bps_per_side), 0.0)
        self.slippage_bps_per_side = max(float(slippage_bps_per_side), 0.0)
        if not self.horizons:
            raise ValueError("Informe pelo menos um horizonte positivo.")
        self.round_trip_cost_pct = 2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side) / 100.0
        self.base_observations = self._build_base_observations()
        self.thresholds = self._build_thresholds()
        self.observations = self._build_profit_observations()

    @staticmethod
    def _number(value: Any):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _prepare_candles(candles):
        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles deve ser um DataFrame.")
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(candles.columns)
        if missing:
            raise ValueError(f"Candles sem colunas obrigatórias: {sorted(missing)}")
        frame = candles.copy().reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype("datetime64[ns]")
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    def _build_base_observations(self):
        result = []
        previous_source_timestamp = None
        previous_oi = None
        for record in self.records:
            metadata = record.market.metadata or {}
            oi = self._number(metadata.get("open_interest"))
            source_timestamp = metadata.get("open_interest_source_timestamp")
            if oi is None or source_timestamp is None:
                continue
            source_key = str(source_timestamp)
            if source_key == previous_source_timestamp:
                continue
            change = self._number(metadata.get("open_interest_change_pct"))
            if change is None and previous_oi not in (None, 0):
                change = ((oi / previous_oi) - 1.0) * 100.0
            previous_source_timestamp = source_key
            previous_oi = oi
            if change is None:
                continue
            result.append({
                "timestamp": pd.Timestamp(record.timestamp),
                "source_timestamp": pd.Timestamp(source_timestamp),
                "decision": record.decision.action,
                "regime": record.market.regime,
                "market_direction": record.market.direction,
                "oi_change_pct": change,
            })
        return result

    def _build_thresholds(self):
        changes = sorted(item["oi_change_pct"] for item in self.base_observations)
        if not changes:
            return {"p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
        return {key: self._percentile(changes, q) for key, q in (("p10", .10), ("p25", .25), ("p75", .75), ("p90", .90))}

    def bucket_for(self, change):
        if change <= self.thresholds["p10"]: return "STRONG_DROP"
        if change <= self.thresholds["p25"]: return "DROP"
        if change < self.thresholds["p75"]: return "NORMAL"
        if change < self.thresholds["p90"]: return "RISE"
        return "STRONG_RISE"

    def _build_profit_observations(self):
        result = []
        for item in self.base_observations:
            future_all = self.candles.loc[self.candles["timestamp"] > item["timestamp"]]
            if future_all.empty: continue
            for horizon in self.horizons:
                future = future_all.head(horizon)
                if len(future) < horizon: continue
                entry_timestamp = pd.Timestamp(future.iloc[0]["timestamp"])
                entry = self._number(future.iloc[0]["open"]); exit_price = self._number(future.iloc[-1]["close"])
                high = self._number(future["high"].max()); low = self._number(future["low"].min())
                if None in (entry, exit_price, high, low) or entry == 0: continue
                future_return = ((exit_price / entry) - 1.0) * 100.0
                decision = str(item["decision"])
                if decision == "LONG":
                    directional = future_return; mfe = ((high / entry) - 1) * 100; mae = ((entry / low) - 1) * 100
                    net = directional - self.round_trip_cost_pct
                elif decision == "SHORT":
                    directional = -future_return; mfe = ((entry / low) - 1) * 100; mae = ((high / entry) - 1) * 100
                    net = directional - self.round_trip_cost_pct
                else:
                    directional = net = 0.0
                    mfe = max(((high / entry) - 1) * 100, ((entry / low) - 1) * 100)
                    mae = min(((high / entry) - 1) * 100, ((entry / low) - 1) * 100)
                result.append(OIProfitObservation(item["timestamp"], entry_timestamp, decision, str(item["regime"]),
                    str(item["market_direction"]), float(item["oi_change_pct"]), self.bucket_for(float(item["oi_change_pct"])),
                    float(entry), horizon, float(future_return), float(directional), float(net), max(float(mfe), 0), max(float(mae), 0)))
        return result

    def summary_by_bucket(self): return self._group(lambda x: (x.horizon_minutes, x.oi_bucket))

    def summary_by_decision_and_bucket(self):
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for x in self.observations: grouped[x.horizon_minutes][x.decision][x.oi_bucket].append(x)
        return {h: {d: {b: self._summarize(v) for b, v in bs.items()} for d, bs in ds.items()} for h, ds in grouped.items()}

    def summary_by_regime_and_bucket(self):
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for x in self.observations: grouped[x.horizon_minutes][x.regime][x.oi_bucket].append(x)
        return {h: {r: {b: self._summarize(v) for b, v in bs.items()} for r, bs in rs.items()} for h, rs in grouped.items()}

    def _group(self, key_fn):
        grouped = defaultdict(lambda: defaultdict(list))
        for x in self.observations:
            h, key = key_fn(x); grouped[h][key].append(x)
        return {h: {k: self._summarize(v) for k, v in ks.items()} for h, ks in grouped.items()}

    @staticmethod
    def _summarize(items):
        if not items: return {"count": 0, "avg_directional_return_pct": 0.0, "avg_net_return_pct": 0.0, "median_net_return_pct": 0.0, "net_positive_rate_pct": 0.0, "avg_mfe_pct": 0.0, "avg_mae_pct": 0.0, "mfe_mae_ratio": 0.0}
        directional = [x.directional_return_pct for x in items]; net = sorted(x.net_return_pct for x in items)
        mfes = [x.mfe_pct for x in items]; maes = [x.mae_pct for x in items]; n = len(items)
        avg_mfe = sum(mfes)/n; avg_mae = sum(maes)/n
        med = net[n//2] if n%2 else (net[n//2-1]+net[n//2])/2
        return {"count": n, "avg_directional_return_pct": round(sum(directional)/n,6), "avg_net_return_pct": round(sum(net)/n,6),
            "median_net_return_pct": round(med,6), "net_positive_rate_pct": round(sum(v>0 for v in net)/n*100,2),
            "avg_mfe_pct": round(avg_mfe,6), "avg_mae_pct": round(avg_mae,6), "mfe_mae_ratio": round(avg_mfe/avg_mae if avg_mae>0 else 0,4)}

    @staticmethod
    def _percentile(values, quantile):
        if len(values)==1: return values[0]
        position=(len(values)-1)*quantile; lower=int(position); upper=min(lower+1,len(values)-1); weight=position-lower
        return values[lower]*(1-weight)+values[upper]*weight
