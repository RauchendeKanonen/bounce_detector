#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
miner_band_lstm.py

End-to-end template to train an LSTM that outputs realistic buy/sell price OFFSETS
for a miners stock using 1h bars and macro drivers (gold, silver, GDX, GDXJ, ES).

- Data: 1H OHLCV where available
- Inputs: [Stock, Gold, Silver, GDX, GDXJ, ES] aligned on 1h
- Outputs: buy_offset, sell_offset relative to current close
- Framework: PyTorch
- Broker: Interactive Brokers (SMART) via ib_insync (recommended)

USAGE (training, offline dummy data):
    python miner_band_lstm.py --ticker AEM --start "2021-01-01" --end "2024-12-31" --epochs 5 --use-dummy

USAGE (training, live IB data):
    python miner_band_lstm.py --ticker AEM --start "2021-01-01" --end "2024-12-31" --client-id 7 --ibg 127.0.0.1 --ibp 7497

USAGE (inference):
    python miner_band_lstm.py --ticker AEM --infer-live --client-id 7
"""

import argparse, math, os, sys
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from numba import typeof
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os, shutil
from pathlib import Path
try: import ta
except: ta=None
from ib_insync import ContFuture, TagValue
# IB imports optional
_IB_OK=True
try:
    from ib_insync import IB, Stock, Future, util, Contract, Index, ContFuture
except:
    _IB_OK=False


def _safe_write_df(df: pd.DataFrame, path: Path, fmt: str = "parquet", compression: str = "snappy"):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt.lower() == "csv":
        df.to_csv(tmp, index=True)
    else:
        # parquet
        df.to_parquet(tmp, index=True, compression=compression)
    tmp.replace(path)  # atomic move

def _load_if_exists(path: Path, fmt: str = "parquet") -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if fmt.lower() == "csv":
        return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return pd.read_parquet(path)

@dataclass
class FetchConfig:
    ticker: str
    start: str
    end: str
    bar_size: str = "1 hour"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    timezone_str: str = "UTC"
#Ambiguous contract: Future(symbol='SI', exchange='COMEX', currency='USD'), possibles are [Contract(secType='FUT', conId=751494443, symbol='SI', lastTradeDateOrContractMonth='20251229', multiplier='1000', exchange='COMEX', currency='USD', localSymbol='SILZ5', tradingClass='SIL')
#Contract(secType='FUT', conId=495512563, symbol='ES', lastTradeDateOrContractMonth='20251219', multiplier='50', exchange='CME', currency='USD', localSymbol='ESZ5', tradingClass='ES')
#Contract(secType='FUT', conId=397594951, symbol='GC', lastTradeDateOrContractMonth='20251229', multiplier='100', exchange='COMEX', currency='USD', localSymbol='GCZ5', tradingClass='GC')

def make_contracts(ticker: str) -> Dict[str,'Contract']:
    stock = Stock(ticker, exchange="SMART", primaryExchange="NYSE", currency="USD")
    gdx = Stock("GDX", exchange="SMART", primaryExchange="ARCA", currency="USD")
    gdxj = Stock("GDXJ", exchange="SMART", primaryExchange="ARCA", currency="USD")
    # Continuous futures
    es = ContFuture("ES", exchange="GLOBEX", currency="USD")
    gc = ContFuture("GC", exchange="COMEX",  currency="USD")
    si = ContFuture("SI", exchange="COMEX",  currency="USD")
    return {"Stock":stock,"GDX":gdx,"GDXJ":gdxj,"ES":es,"GC":gc,"SI":si}



def fetch_ib_data(
    ib: 'IB',
    contracts: Dict[str, 'Contract'],
    cfg: FetchConfig,
    out_dir: Optional[str] = None,
    save_fmt: str = "parquet",           # "parquet" or "csv"
    chunk_days: int = 180,               # split to avoid IB timeouts
    resume: bool = True,                 # append to existing files
    compression: str = "snappy",         # parquet compression
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical bars in chunks and optionally save each instrument to disk.
    Returns a dict of DataFrames regardless of saving.

    File layout (if out_dir is provided):
      out_dir/
        STOCK.parquet
        GDX.parquet
        GDXJ.parquet
        ES.parquet
        GC.parquet
        SI.parquet
    """

    chart_opts = [TagValue("continuous", "2"),  # 1=calendar, 2=nearest, 3=custom
                  TagValue("adjusted", "true")]  # back-adjust for rolls

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_one(contract: 'Contract', key: str) -> pd.DataFrame:
        # Decide file path
        print(f"fetching {Contract.symbol}")
        fpath = None
        if out_dir:
            ext = ".csv" if save_fmt.lower() == "csv" else ".parquet"
            fpath = out_dir / f"{key}{ext}"

        # Determine start/end in UTC
        start_ts = pd.Timestamp(cfg.start, tz="UTC")
        end_ts   = pd.Timestamp(cfg.end,   tz="UTC") + pd.Timedelta(days=1)  # inclusive end

        # Try to resume from existing file
        cached = None
        if resume and fpath and fpath.exists():
            try:
                cached = _load_if_exists(fpath, save_fmt)
                # Ensure tz-aware UTC
                cached.index = pd.to_datetime(cached.index, utc=True)
                if len(cached):
                    last = cached.index.max()
                    # Move start forward, but keep overlap of 1 bar for de-dupe
                    start_ts = max(start_ts, last - pd.Timedelta(hours=1))
            except Exception as e:
                print(f"[WARN] Failed to load cache for {key}: {e}")

        # Chunked pulls
        all_bars: List = []
        cursor = start_ts
        while cursor < end_ts:
            window_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
            try:

                    chart_opts = [TagValue("continuous", "2"),  # 1=calendar, 2=nearest
                                  TagValue("adjusted", "true")]
                    if contract.symbol == "SI" or contract.symbol == "GC" or contract.symbol == "ES":
                        bars = ib.reqHistoricalData(
                            contract=contract,
                            endDateTime=window_end,
                            durationStr=f"{(window_end - cursor).days or 1} D",
                            barSizeSetting="1 hour",
                            whatToShow="TRADES",
                            useRTH=False,
                            formatDate=2,
                            keepUpToDate=False,
                            chartOptions=chart_opts,
                        )
                    else:
                        bars = ib.reqHistoricalData(
                            contract=contract,
                            endDateTime=window_end,
                            durationStr=f"{(window_end - cursor).days or 1} D",
                            barSizeSetting=cfg.bar_size,
                            whatToShow=cfg.what_to_show,
                            useRTH=cfg.use_rth,
                            formatDate=2,
                        )
            except Exception as e:
                print(f"[ERROR] reqHistoricalData failed for {key} at {cursor}→{window_end}: {e}")
                # advance a bit to avoid infinite loop on repeated failure
                cursor = window_end
                continue

            if not bars:
                # No data; still advance
                cursor = window_end
                continue

            all_bars.extend(bars)
            cursor = window_end
            time.sleep(1)
            print(f"got {len(bars)} Bars")

        # Convert to DataFrame
        if not all_bars:
            print(f"[WARN] No data received for {key}")
            # Return cached if we have it
            if cached is not None:
                return cached
            return pd.DataFrame(columns=["open","high","low","close","volume"])

        df = util.df(all_bars)
        if df is None or len(df) == 0:
            print(f"[WARN] Empty conversion for {key}")
            return cached if cached is not None else pd.DataFrame(columns=["open","high","low","close","volume"])

        df.rename(columns={"date":"timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[["open","high","low","close","volume"]].sort_index()

        # Merge with cache and de-duplicate
        if cached is not None and len(cached):
            merged = pd.concat([cached, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = df

        # Persist
        if fpath:
            try:
                _safe_write_df(merged, fpath, fmt=save_fmt, compression=compression)
                print(f"[OK] Saved {key} → {fpath}  ({len(merged)} rows)")
            except Exception as e:
                print(f"[ERROR] Save failed for {key}: {e}")

        return merged

    out: Dict[str, pd.DataFrame] = {}
    for key, contract in contracts.items():
        try:
            # Always qualify to avoid SMART confusion/timeouts
            ib.qualifyContracts(contract)
            out[key] = _fetch_one(contract, key)
        except Exception as e:
            print(f"[ERROR] Fetch failed for {key}: {e}")

    return out



# dummy data generator
def dummy_series(name,start,end,freq="1H"):
    idx=pd.date_range(start,end,freq=freq,tz="UTC",inclusive="left")
    n=len(idx); rng=np.random.default_rng(abs(hash(name))%(2**32))
    price=100+np.cumsum(rng.normal(0,0.2,n))
    price=pd.Series(price).rolling(24,min_periods=1).mean().values+rng.normal(0,0.3,n)
    high=price+rng.normal(0.1,0.08,n).clip(0,None)
    low=price-rng.normal(0.1,0.08,n).clip(0,None)
    open_=price+rng.normal(0,0.05,n); close=price+rng.normal(0,0.05,n)
    vol=rng.lognormal(11,0.5,n)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol},index=idx)

def dummy_data(start,end):
    return {n:dummy_series(n,start,end) for n in ["STOCK","GDX","GDXJ","ES","GC","SI"]}


def merge_align(data:Dict[str,pd.DataFrame]):
    all_idx=None
    for df in data.values():
        all_idx=df.index if all_idx is None else all_idx.union(df.index)
    all_idx=all_idx.sort_values()
    outs=[]
    for k,df in data.items():
        df=df.reindex(all_idx).ffill(limit=2)
        outs.append(df.add_prefix(f"{k}_"))
    return pd.concat(outs,axis=1).dropna()


def add_ind(df):
    out=df.copy()
    keys=["STOCK","GDX","GDXJ","ES","GC","SI"]
    for k in keys:
        c=f"{k}_close"; v=f"{k}_volume"
        if c in out: out[f"{k}_ret1"]=out[c].pct_change()
        if c in out:
            out[f"{k}_z20"]=(out[c]-out[c].rolling(20).mean())/(out[c].rolling(20).std()+1e-9)
        if v in out:
            out[f"{k}_v_z20"]=(np.log1p(out[v])-np.log1p(out[v]).rolling(20).mean())/(np.log1p(out[v]).rolling(20).std()+1e-9)
    return out.dropna()


def label_offsets(df,forward=10,stock="STOCK"):
    c=f"{stock}_close"; h=f"{stock}_high"; l=f"{stock}_low"; v=f"{stock}_volume"
    work=df[[c,h,l,v]].dropna()
    N=len(work); buy=np.full(N,np.nan); sell=np.full(N,np.nan)
    vol=work[v].values; volth=np.nanquantile(vol,0.2)
    closes=work[c].values; highs=work[h].values; lows=work[l].values
    for i in range(N-forward):
        close_t=closes[i]
        mask=vol[i+1:i+1+forward]>=volth
        wl=lows[i+1:i+1+forward][mask]; wh=highs[i+1:i+1+forward][mask]
        if len(wl)>0:
            fill_low=np.nanpercentile(wl,25)
            buy[i]=(fill_low-close_t)/close_t
        if len(wh)>0:
            fill_high=np.nanpercentile(wh,75)
            sell[i]=(fill_high-close_t)/close_t
    lab=pd.DataFrame({"buy_offset":buy,"sell_offset":sell},index=work.index)
    return df.join(lab).dropna(subset=["buy_offset","sell_offset"])


class SeqDS(Dataset):
    def __init__(self,df,seq=60):
        self.seq=seq
        self.y=df[["buy_offset","sell_offset"]].values.astype(np.float32)
        self.x=df.drop(columns=["buy_offset","sell_offset"]).values.astype(np.float32)
        self.idx=[i for i in range(seq,len(df))]
        self.mean=self.x.mean(0); self.std=self.x.std(0)+1e-6
        self.x=(self.x-self.mean)/self.std
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        j=self.idx[i]; return torch.tensor(self.x[j-self.seq:j]), torch.tensor(self.y[j])


class LSTM(nn.Module):
    def __init__(self,f,h=128,layers=2,drop=0.2):
        super().__init__()
        self.l=nn.LSTM(f,h,layers,batch_first=True,dropout=drop*(layers>1))
        self.hd=nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(drop),nn.Linear(h//2,2))
    def forward(self,x):
        o,_=self.l(x); return self.hd(o[:,-1])


def train(df,seq,epochs=5,bs=64,lr=1e-3,device="cpu"):
    n=len(df); sp=int(n*0.8)
    tr,va=df.iloc[:sp],df.iloc[sp-seq:]
    dtr, dva = SeqDS(tr,seq), SeqDS(va,seq)
    trl=DataLoader(dtr,bs,True,drop_last=True)
    val=DataLoader(dva,bs,False)
    m=LSTM(dtr.x.shape[1]).to(device); opt=torch.optim.AdamW(m.parameters(),lr=lr)
    lossf=nn.L1Loss(); best=1e9
    for e in range(1,epochs+1):
        m.train(); tl=0
        for xb,yb in trl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(); p=m(xb); l=lossf(p,yb); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
            tl+=l.item()*xb.size(0)
        tl/=len(dtr)
        m.eval(); vl=0
        with torch.no_grad():
            for xb,yb in val:
                xb,yb=xb.to(device),yb.to(device); p=m(xb); vl+=lossf(p,yb).item()*xb.size(0)
        vl/=len(dva)
        print(f"Epoch {e}: train={tl:.4f} val={vl:.4f}")
        if vl<best:
            best=vl
            torch.save({"model":m.state_dict(),"f":dtr.x.shape[1],"seq":seq,"mean":dtr.mean,"std":dtr.std,"cols":df.drop(columns=['buy_offset','sell_offset']).columns.tolist()},"band_lstm_best.pth")


def infer(df,device="cpu"):
    ck=torch.load("band_lstm_best.pth",map_location=device)
    m=LSTM(ck["f"]); m.load_state_dict(ck["model"]); m.to(device); m.eval()
    x=df[ck["cols"]].values.astype(np.float32)
    x=(x-ck["mean"])/ck["std"]
    x=torch.tensor(x[-ck["seq"]:, :]).unsqueeze(0).to(device)
    p=m(x).cpu().numpy()[0]
    close=df["STOCK_close"].iloc[-1]
    return close*(1+p[0]), close*(1+p[1]), p


def main():
    A=argparse.ArgumentParser()
    A.add_argument("--ticker",default="AEM")
    A.add_argument("--start",default="2021-01-01")
    A.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    A.add_argument("--seq",type=int,default=60)
    A.add_argument("--forward",type=int,default=10)
    A.add_argument("--epochs",type=int,default=5)
    A.add_argument("--use-dummy",action="store_true")
    A.add_argument("--infer-live",action="store_true")
    A.add_argument("--device",default="cpu")
    a=A.parse_args()

    if a.use_dummy:
        raw=dummy_data(a.start,a.end)
    else:
        ib=IB(); ib.connect("127.0.0.1",7497, clientId=6658, readonly=True)
        contracts = make_contracts(a.ticker)  # keep your existing function
        cfg = FetchConfig(ticker=a.ticker, start=a.start, end=a.end)

        raw = fetch_ib_data(
            ib,
            contracts,
            cfg,
            out_dir="data_1h_utc",  # folder to save files
            save_fmt="parquet",  # or "csv"
            chunk_days=180,  # safer for 1h bars
            resume=True,  # append to existing
            compression="snappy",  # parquet compression
        )





    df=merge_align(raw)
    df=add_ind(df)
    df=label_offsets(df,a.forward)

    if not a.infer_live:
        train(df,a.seq,a.epochs,device=a.device)
        print("✅ training done. model saved.")
    else:
        buy,sell,_=infer(df,a.device)
        print(f"Buy @ {buy:.2f}   Sell @ {sell:.2f}")

if __name__=="__main__":
    main()
