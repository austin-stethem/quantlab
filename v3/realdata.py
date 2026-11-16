"""Real-data connectors for v3. Raw files land in qlraw/, every file is hashed into MANIFEST.csv.

Needs two environment variables, set by the owner, never written into code:
    SEC_UA     "Name contact-email"  (SEC fair-access policy requires a declared contact)
    EODHD_KEY  EODHD API token
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parents[2] / "qlraw"
MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "MANIFEST.csv"


# ----------------------------------------------------------------------------- plumbing
def _get(url: str, headers: dict, pause: float) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            time.sleep(pause)
            return data
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url.split('api_token')[0]}")


def save(rel: str, data: bytes, source: str) -> pathlib.Path:
    p = RAW / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    row = pd.DataFrame([{"file": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                         "source": source, "fetched": pd.Timestamp.now(tz="America/Chicago").isoformat()}])
    row.to_csv(MANIFEST, mode="a", header=not MANIFEST.exists(), index=False)
    return p


# ----------------------------------------------------------------------------- SEC EDGAR
class Edgar:
    def __init__(self):
        ua = os.environ.get("SEC_UA", "").strip()
        if "@" not in ua:
            raise SystemExit("SEC_UA is not set. The SEC requires a declared contact; see README.")
        self.h = {"User-Agent": ua, "Accept-Encoding": "identity"}

    def submissions(self, cik: int) -> dict:
        rel = f"edgar/submissions/CIK{cik:010d}.json"
        if (RAW / rel).exists():
            return json.loads((RAW / rel).read_text())
        d = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", self.h, 0.12)
        save(rel, d, "sec submissions")
        js = json.loads(d)
        for f in js.get("filings", {}).get("files", []):          # older filings are paged
            r2 = f"edgar/submissions/{f['name']}"
            if not (RAW / r2).exists():
                save(r2, _get(f"https://data.sec.gov/submissions/{f['name']}", self.h, 0.12), "sec submissions page")
        return js

    def companyfacts(self, cik: int) -> dict:
        rel = f"edgar/companyfacts/CIK{cik:010d}.json"
        if not (RAW / rel).exists():
            save(rel, _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", self.h, 0.12), "sec companyfacts")
        return json.loads((RAW / rel).read_text())


def filings_table(js: dict, cik: int) -> pd.DataFrame:
    """All filings for one CIK (recent + paged), with acceptance timestamps."""
    blocks = [js["filings"]["recent"]]
    for f in js.get("filings", {}).get("files", []):
        p = RAW / "edgar" / "submissions" / f["name"]
        if p.exists():
            blocks.append(json.loads(p.read_text()))
    df = pd.concat([pd.DataFrame(b) for b in blocks], ignore_index=True)
    df["cik"] = cik
    # acceptanceDateTime carries a 'Z' but EDGAR stamps Eastern time; the dating audit verifies this choice.
    df["accepted"] = pd.to_datetime(df["acceptanceDateTime"].str.slice(0, 19))
    return df


def events_and_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    k8 = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.contains(r"\b2\.02\b")]
    per = df[df["form"].isin(["10-Q", "10-K", "10-KT"])].copy()
    per = per.rename(columns={"reportDate": "period_end", "filingDate": "filed"})
    per["period_end"] = pd.to_datetime(per["period_end"], errors="coerce")
    per["filed"] = pd.to_datetime(per["filed"])
    return (k8[["cik", "accepted", "form", "accessionNumber", "primaryDocument"]],
            per[["cik", "period_end", "filed", "form"]].dropna())


def xbrl_series(facts: dict, cik: int) -> pd.DataFrame:
    """Point-in-time EPS (diluted), stockholders' equity and shares, each with its filing date."""
    out = []
    want = {("us-gaap", "EarningsPerShareDiluted"): "eps", ("us-gaap", "StockholdersEquity"): "equity",
            ("dei", "EntityCommonStockSharesOutstanding"): "shares"}
    for (ns, tag), name in want.items():
        units = facts.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {})
        for u, rows in units.items():
            for r in rows:
                out.append({"cik": cik, "field": name, "end": r.get("end"), "filed": r.get("filed"), "form": r.get("form"),
                            "fp": r.get("fp"), "value": r.get("val")})
    df = pd.DataFrame(out)
    if len(df):
        df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"])
    return df


# ----------------------------------------------------------------------------- EODHD prices
class Eodhd:
    def __init__(self, key: str | None = None):
        self.key = key or os.environ.get("EODHD_KEY", "").strip()
        if not self.key:
            raise SystemExit("EODHD_KEY is not set.")

    def _json(self, path: str, rel: str, params: str = "") -> list:
        if not (RAW / rel).exists():
            save(rel, _get(f"https://eodhd.com/api/{path}?api_token={self.key}&fmt=json{params}", {}, 0.05), "eodhd " + path.split("/")[0])
        return json.loads((RAW / rel).read_text())

    def symbols(self, delisted: bool) -> pd.DataFrame:
        return pd.DataFrame(self._json("exchange-symbol-list/US", f"eodhd/symbols_{'delisted' if delisted else 'active'}.json",
                                       "&delisted=1" if delisted else ""))

    def bars(self, sym: str) -> pd.DataFrame:
        return pd.DataFrame(self._json(f"eod/{sym}.US", f"eodhd/eod/{sym}.json", "&from=2003-01-01"))

    def dividends(self, sym: str) -> pd.DataFrame:
        return pd.DataFrame(self._json(f"div/{sym}.US", f"eodhd/div/{sym}.json", "&from=2003-01-01"))

    def splits(self, sym: str) -> pd.DataFrame:
        return pd.DataFrame(self._json(f"splits/{sym}.US", f"eodhd/splits/{sym}.json", "&from=2003-01-01"))


def total_return(bars: pd.DataFrame, divs: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Daily total return built from RAW closes, split ratios and cash dividends (ex-date)."""
    b = bars[["date", "close", "adjusted_close"]].copy()
    b["date"] = pd.to_datetime(b["date"]); b = b.set_index("date").sort_index()
    factor = pd.Series(1.0, index=b.index)
    if len(splits):
        for _, s in splits.iterrows():
            num, den = (float(x) for x in str(s["split"]).split("/"))
            d = pd.Timestamp(s["date"])
            if d in factor.index:
                factor.loc[d] = num / den
    div = pd.Series(0.0, index=b.index)
    if len(divs):
        for _, d in divs.iterrows():
            t = pd.Timestamp(d["date"])
            if t in div.index:
                div.loc[t] += float(d.get("unadjustedValue", d.get("value", 0.0)) or 0.0)
    prev = b["close"].shift(1)
    b["ret"] = (b["close"] * factor + div) / prev - 1
    b["ret_vendor"] = b["adjusted_close"].pct_change()
    b["diff"] = (b["ret"] - b["ret_vendor"]).abs()
    return b


# ----------------------------------------------------------------------------- factors
def french() -> pd.DataFrame:
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    frames = []
    for name in ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"):
        p = RAW / "french" / name
        if not p.exists():
            save(f"french/{name}", _get(base + name, {"User-Agent": "Mozilla/5.0"}, 0.5), "ken french library")
        with zipfile.ZipFile(p) as z:
            txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        start = next(i for i, l in enumerate(txt) if l.strip()[:8].isdigit())
        rows = [l.split(",") for l in txt[start:] if l.strip()[:8].isdigit()]
        hdr = [h.strip() for h in txt[start - 1].split(",")]
        df = pd.DataFrame(rows, columns=["date"] + hdr[1:]).set_index("date")
        frames.append(df.astype(float) / 100)
    ff = pd.concat(frames, axis=1, join="inner")
    ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
    return ff.rename(columns=lambda c: c.strip())


def load_market():
    raise SystemExit("Real market assembly runs after the EDGAR and EODHD pulls; see README step list.")
