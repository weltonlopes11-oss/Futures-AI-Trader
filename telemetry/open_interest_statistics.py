from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite
from statistics import mean, median
from typing import Any


class OpenInterestStatistics:
    """Estatísticas exploratórias por observação oficial de Open Interest.

    Datasets novos usam o timestamp oficial da observação como identidade.
    Datasets legados sem esse campo mantêm compatibilidade por mudança de valor.
    """

    def __init__(self, records):
        self.records = records
        self.observations = self._build_observations()

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _direction(value: float) -> str:
        if value > 0: return "UP"
        if value < 0: return "DOWN"
        return "STABLE"

    def _build_observations(self):
        observations=[]; previous_source_timestamp=None; previous_oi=None; previous_price=None
        for record in self.records:
            metadata=record.market.metadata or {}
            oi=self._number(metadata.get("open_interest"))
            if oi is None: continue
            source_timestamp=metadata.get("open_interest_source_timestamp")
            if source_timestamp is not None:
                source_key=str(source_timestamp)
                if source_key==previous_source_timestamp: continue
            else:
                source_key=None
                if previous_oi is not None and oi==previous_oi: continue

            price=self._number(record.close_price); api_change=self._number(metadata.get("open_interest_change_pct"))
            if previous_oi is None:
                oi_change_pct=api_change; price_change_pct=None
            else:
                oi_change_pct=((oi/previous_oi)-1.0)*100.0 if previous_oi!=0 else None
                price_change_pct=((price/previous_price)-1.0)*100.0 if price is not None and previous_price not in (None,0) else None
            observations.append({"timestamp":record.timestamp,"source_timestamp":source_timestamp,"decision":record.decision.action,
                "regime":record.market.regime,"market_direction":record.market.direction,"open_interest":oi,
                "oi_change_pct":oi_change_pct,"price":price,"price_change_pct":price_change_pct})
            previous_source_timestamp=source_key; previous_oi=oi
            if price is not None: previous_price=price
        return observations

    @property
    def total_observations(self): return len(self.observations)

    def oi_direction_distribution(self):
        c=Counter()
        for x in self.observations:
            if x["oi_change_pct"] is not None: c[self._direction(x["oi_change_pct"])]+=1
        return dict(c)

    def by_decision(self):
        r=defaultdict(Counter)
        for x in self.observations:
            if x["oi_change_pct"] is not None: r[x["decision"]][self._direction(x["oi_change_pct"])]+=1
        return {k:dict(v) for k,v in r.items()}

    def by_regime(self):
        r=defaultdict(Counter)
        for x in self.observations:
            if x["oi_change_pct"] is not None: r[x["regime"]][self._direction(x["oi_change_pct"])]+=1
        return {k:dict(v) for k,v in r.items()}

    def price_oi_combinations(self):
        c=Counter()
        for x in self.observations:
            oi,p=x["oi_change_pct"],x["price_change_pct"]
            if oi is not None and p is not None: c[f"PRICE_{self._direction(p)}__OI_{self._direction(oi)}"]+=1
        return dict(c)

    def change_statistics(self):
        return {"oi":self._describe(x["oi_change_pct"] for x in self.observations),"price":self._describe(x["price_change_pct"] for x in self.observations)}

    def decision_change_statistics(self):
        g=defaultdict(list)
        for x in self.observations:
            if x["oi_change_pct"] is not None: g[x["decision"]].append(x["oi_change_pct"])
        return {k:self._describe(v) for k,v in g.items()}

    @classmethod
    def _describe(cls,values):
        clean=sorted(value for raw in values if (value:=cls._number(raw)) is not None)
        if not clean: return {"count":0,"average":0.0,"median":0.0,"min":0.0,"p10":0.0,"p25":0.0,"p75":0.0,"p90":0.0,"max":0.0}
        return {"count":len(clean),"average":round(mean(clean),6),"median":round(median(clean),6),"min":round(clean[0],6),
            "p10":round(cls._percentile(clean,.10),6),"p25":round(cls._percentile(clean,.25),6),"p75":round(cls._percentile(clean,.75),6),
            "p90":round(cls._percentile(clean,.90),6),"max":round(clean[-1],6)}

    @staticmethod
    def _percentile(values,quantile):
        if len(values)==1:return values[0]
        pos=(len(values)-1)*quantile; lo=int(pos); hi=min(lo+1,len(values)-1); w=pos-lo
        return values[lo]*(1-w)+values[hi]*w
