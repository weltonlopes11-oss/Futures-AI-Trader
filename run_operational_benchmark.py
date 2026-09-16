from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.binance_funding_rate import BinanceFundingRateLoader
from backtest.binance_metrics_data_vision import BinanceMetricsDataVisionLoader
from backtest.operational_backtest import OperationalBacktest


def atr(frame, period=14):
    prev = frame["close"].shift(1)
    tr = pd.concat([frame["high"]-frame["low"], (frame["high"]-prev).abs(), (frame["low"]-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def trend(frame, fast=12, slow=26):
    f = frame["close"].ewm(span=fast, adjust=False).mean(); s = frame["close"].ewm(span=slow, adjust=False).mean()
    return pd.Series(pd.NA, index=frame.index).mask(f > s, "LONG").mask(f < s, "SHORT")


def causal_context(signal, higher, label):
    h = higher.copy(); h[label] = trend(h); h["available_at"] = pd.to_datetime(h["close_time"], errors="coerce")
    return pd.merge_asof(signal.sort_values("timestamp"), h[["available_at", label]].dropna().sort_values("available_at"), left_on="timestamp", right_on="available_at", direction="backward")


def require_coverage(frame, expected, label):
    if len(frame) < int(expected * .99): raise RuntimeError(f"Insufficient {label} coverage: {len(frame)}/{expected}")


def build_decisions(context, use_oi=False, use_funding=False):
    c = context.copy(); c["decision"] = "NO_TRADE"
    base_long = (c.signal_15m == "LONG") & (c.trend_1h == "LONG") & (c.trend_4h == "LONG")
    base_short = (c.signal_15m == "SHORT") & (c.trend_1h == "SHORT") & (c.trend_4h == "SHORT")

    # OI confirmation: only open a directional trade when aggregate open positions
    # are expanding versus the previous official observation. OI falling is treated
    # as closing/deleveraging and does not confirm a new directional position.
    oi_confirm = c.open_interest_change_pct > 0 if use_oi else pd.Series(True, index=c.index)

    # Funding anti-crowding filter. Funding is aligned backward using only the last
    # funding observation already published at decision time. LONG avoids periods
    # where longs are paying shorts; SHORT avoids periods where shorts are paying longs.
    if use_funding:
        funding_long = c.funding_rate <= 0
        funding_short = c.funding_rate >= 0
    else:
        funding_long = pd.Series(True, index=c.index)
        funding_short = pd.Series(True, index=c.index)

    c.loc[base_long & oi_confirm & funding_long, "decision"] = "LONG"
    c.loc[base_short & oi_confirm & funding_short, "decision"] = "SHORT"
    return c


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    start = datetime.fromisoformat(os.getenv("BACKTEST_START_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00")); end = datetime.fromisoformat(os.getenv("BACKTEST_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if start.tzinfo is None: start=start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None: end=end.replace(tzinfo=timezone.utc)
    prices=BinanceDataVisionLoader(); metrics=BinanceMetricsDataVisionLoader(); funding_loader=BinanceFundingRateLoader()
    m15=prices.fetch_window(symbol,"15m",start,end); h1=prices.fetch_window(symbol,"1h",start,end); h4=prices.fetch_window(symbol,"4h",start,end); oi=metrics.fetch_window(symbol,start,end); funding=funding_loader.fetch_window(symbol,start,end)
    mins=int((end-start).total_seconds()//60); require_coverage(m15,mins//15,"15m"); require_coverage(h1,mins//60,"1h"); require_coverage(h4,mins//240,"4h")
    m15["atr"]=atr(m15); m15["signal_15m"]=trend(m15); context=causal_context(m15,h1,"trend_1h"); context=causal_context(context,h4,"trend_4h"); context=metrics.align_causally(context,oi); context=funding_loader.align_causally(context,funding)
    oi_coverage=context.open_interest.notna().mean()*100
    funding_coverage=context.funding_rate.notna().mean()*100
    if oi_coverage < 99: raise RuntimeError(f"Insufficient OI coverage: {oi_coverage:.2f}%")
    if funding_coverage < 99: raise RuntimeError(f"Insufficient funding coverage: {funding_coverage:.2f}%")
    engine=OperationalBacktest(fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE","4")),slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE","1")))
    out=Path("artifacts/operational-benchmark"); out.mkdir(parents=True,exist_ok=True); rows=[]
    modes=(("baseline",False,False),("open_interest",True,False),("oi_funding",True,True))
    for mode,use_oi,use_funding in modes:
        decisions=build_decisions(context,use_oi=use_oi,use_funding=use_funding)
        for rr in (1.,1.5,2.,3.):
            trades=engine.run(decisions,decisions[["timestamp","decision"]],rr=rr); met=engine.metrics(trades); met.update({"mode":mode,"rr":rr}); rows.append(met); trades.to_csv(out/f"trades_{mode}_rr_{rr}.csv",index=False)
    results=pd.DataFrame(rows)[["mode","rr","trades","win_rate_pct","net_return_pct","profit_factor","expectancy_pct","max_drawdown_pct","payoff","long","short"]]
    results.to_csv(out/"results.csv",index=False)
    manifest={"source":"Binance Data Vision + Binance USD-M public REST","market":"USD-M Futures","symbol":symbol,"start_utc":start.isoformat(),"end_utc":end.isoformat(),"signal":"15m","structure":"1h","regime":"4h","open_interest":"Binance USD-M futures metrics archive, causal backward alignment","oi_rule":"open_interest_change_pct > 0","oi_observations":len(oi),"oi_coverage_pct":oi_coverage,"funding_rate":"Binance /fapi/v1/fundingRate, causal backward alignment","funding_rule":"with OI expansion: LONG funding_rate <= 0; SHORT funding_rate >= 0","funding_observations":len(funding),"funding_coverage_pct":funding_coverage,"fee_bps_per_side":engine.fee_bps_per_side,"slippage_bps_per_side":engine.slippage_bps_per_side,"candles_15m":len(m15),"candles_1h":len(h1),"candles_4h":len(h4)}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2)); print(results.to_string(index=False))

if __name__ == "__main__": main()
