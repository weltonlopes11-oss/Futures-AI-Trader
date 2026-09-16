from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable
import pandas as pd

@dataclass(slots=True)
class OIProfitObservation:
    signal_timestamp: pd.Timestamp; entry_timestamp: pd.Timestamp; decision: str; regime: str; market_direction: str
    oi_change_pct: float; oi_bucket: str; entry_price: float; horizon_minutes: int; future_return_pct: float
    directional_return_pct: float; net_return_pct: float; mfe_pct: float; mae_pct: float

class OIProfitAnalyzer:
    DEFAULT_HORIZONS=(5,15,30,60)
    def __init__(self,records:Iterable,candles:pd.DataFrame,horizons=DEFAULT_HORIZONS,fee_bps_per_side:float=0.0,slippage_bps_per_side:float=0.0):
        self.records=list(records); self.candles=self._prepare_candles(candles)
        self.horizons=tuple(sorted(set(int(x) for x in horizons if int(x)>0)))
        self.fee_bps_per_side=max(float(fee_bps_per_side),0.0); self.slippage_bps_per_side=max(float(slippage_bps_per_side),0.0)
        if not self.horizons: raise ValueError("Informe pelo menos um horizonte positivo.")
        self.round_trip_cost_pct=2.0*(self.fee_bps_per_side+self.slippage_bps_per_side)/100.0
        self.base_observations=self._build_base_observations(); self.thresholds=self._build_thresholds(); self.observations=self._build_profit_observations()
    @staticmethod
    def _number(value:Any):
        try:number=float(value)
        except (TypeError,ValueError):return None
        return number if isfinite(number) else None
    @staticmethod
    def _prepare_candles(candles):
        if not isinstance(candles,pd.DataFrame):raise TypeError("candles deve ser um DataFrame.")
        required={"timestamp","open","high","low","close"}; missing=required-set(candles.columns)
        if missing:raise ValueError(f"Candles sem colunas obrigatórias: {sorted(missing)}")
        frame=candles.copy().reset_index(drop=True); frame["timestamp"]=pd.to_datetime(frame["timestamp"]).astype("datetime64[ns]")
        for c in ("open","high","low","close"):frame[c]=pd.to_numeric(frame[c],errors="raise")
        return frame.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    def _build_base_observations(self):
        result=[]; previous_source_timestamp=None; previous_oi=None
        for record in self.records:
            metadata=record.market.metadata or {}; oi=self._number(metadata.get("open_interest"))
            if oi is None:continue
            source_timestamp=metadata.get("open_interest_source_timestamp")
            if source_timestamp is not None:
                source_key=str(source_timestamp)
                if source_key==previous_source_timestamp:continue
            else:
                source_key=None
                if previous_oi is not None and oi==previous_oi:continue
            change=self._number(metadata.get("open_interest_change_pct"))
            if change is None and previous_oi not in (None,0):change=((oi/previous_oi)-1.0)*100.0
            previous_source_timestamp=source_key; previous_oi=oi
            if change is None:continue
            result.append({"timestamp":pd.Timestamp(record.timestamp),"source_timestamp":pd.Timestamp(source_timestamp) if source_timestamp is not None else None,
                "decision":record.decision.action,"regime":record.market.regime,"market_direction":record.market.direction,"oi_change_pct":change})
        return result
    def _build_thresholds(self):
        changes=sorted(x["oi_change_pct"] for x in self.base_observations)
        if not changes:return {"p10":0.0,"p25":0.0,"p75":0.0,"p90":0.0}
        return {k:self._percentile(changes,q) for k,q in (("p10",.10),("p25",.25),("p75",.75),("p90",.90))}
    def bucket_for(self,c):
        if c<=self.thresholds["p10"]:return "STRONG_DROP"
        if c<=self.thresholds["p25"]:return "DROP"
        if c<self.thresholds["p75"]:return "NORMAL"
        if c<self.thresholds["p90"]:return "RISE"
        return "STRONG_RISE"
    def _build_profit_observations(self):
        result=[]
        for item in self.base_observations:
            future_all=self.candles.loc[self.candles["timestamp"]>item["timestamp"]]
            for horizon in self.horizons:
                future=future_all.head(horizon)
                if len(future)<horizon:continue
                et=pd.Timestamp(future.iloc[0]["timestamp"]); entry=self._number(future.iloc[0]["open"]); exitp=self._number(future.iloc[-1]["close"]); high=self._number(future["high"].max()); low=self._number(future["low"].min())
                if None in (entry,exitp,high,low) or entry==0:continue
                fr=((exitp/entry)-1)*100; d=str(item["decision"])
                if d=="LONG":directional=fr; mfe=((high/entry)-1)*100; mae=((entry/low)-1)*100; net=directional-self.round_trip_cost_pct
                elif d=="SHORT":directional=-fr; mfe=((entry/low)-1)*100; mae=((high/entry)-1)*100; net=directional-self.round_trip_cost_pct
                else:directional=net=0.0; mfe=max(((high/entry)-1)*100,((entry/low)-1)*100); mae=min(((high/entry)-1)*100,((entry/low)-1)*100)
                result.append(OIProfitObservation(item["timestamp"],et,d,str(item["regime"]),str(item["market_direction"]),float(item["oi_change_pct"]),self.bucket_for(float(item["oi_change_pct"])),float(entry),horizon,float(fr),float(directional),float(net),max(float(mfe),0),max(float(mae),0)))
        return result
    def summary_by_bucket(self):return self._group(lambda x:(x.horizon_minutes,x.oi_bucket))
    def summary_by_decision_and_bucket(self):
        g=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for x in self.observations:g[x.horizon_minutes][x.decision][x.oi_bucket].append(x)
        return {h:{d:{b:self._summarize(v) for b,v in bs.items()} for d,bs in ds.items()} for h,ds in g.items()}
    def summary_by_regime_and_bucket(self):
        g=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for x in self.observations:g[x.horizon_minutes][x.regime][x.oi_bucket].append(x)
        return {h:{r:{b:self._summarize(v) for b,v in bs.items()} for r,bs in rs.items()} for h,rs in g.items()}
    def _group(self,key_fn):
        g=defaultdict(lambda:defaultdict(list))
        for x in self.observations:h,k=key_fn(x);g[h][k].append(x)
        return {h:{k:self._summarize(v) for k,v in ks.items()} for h,ks in g.items()}
    @staticmethod
    def _summarize(items):
        if not items:return {"count":0,"avg_directional_return_pct":0.0,"avg_net_return_pct":0.0,"median_net_return_pct":0.0,"net_positive_rate_pct":0.0,"avg_mfe_pct":0.0,"avg_mae_pct":0.0,"mfe_mae_ratio":0.0}
        directional=[x.directional_return_pct for x in items];net=sorted(x.net_return_pct for x in items);mfes=[x.mfe_pct for x in items];maes=[x.mae_pct for x in items];n=len(items);am=sum(mfes)/n;aa=sum(maes)/n;med=net[n//2] if n%2 else (net[n//2-1]+net[n//2])/2
        return {"count":n,"avg_directional_return_pct":round(sum(directional)/n,6),"avg_net_return_pct":round(sum(net)/n,6),"median_net_return_pct":round(med,6),"net_positive_rate_pct":round(sum(v>0 for v in net)/n*100,2),"avg_mfe_pct":round(am,6),"avg_mae_pct":round(aa,6),"mfe_mae_ratio":round(am/aa if aa>0 else 0,4)}
    @staticmethod
    def _percentile(values,q):
        if len(values)==1:return values[0]
        p=(len(values)-1)*q;l=int(p);u=min(l+1,len(values)-1);w=p-l
        return values[l]*(1-w)+values[u]*w
