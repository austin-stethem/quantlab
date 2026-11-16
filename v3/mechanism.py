"""Does the drift get stronger exactly where attention is thinner?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

PRED = {"friday": +1, "busy": +1, "instown": -1, "coverage": -1}


def attention_vars(m, ev: pd.DataFrame) -> pd.DataFrame:
    e = ev.copy()
    e["friday"] = (pd.to_datetime(e["accepted"]).dt.weekday == 4).astype(float)
    n = e.groupby("day0")["cik"].transform("size")
    e["busy"] = (np.log(n) - np.log(n).mean()) / np.log(n).std()
    if m.attention is not None and len(m.attention):
        a = m.attention.sort_values("date")
        e["_d"] = m.days[e["day0"].values]
        e = pd.merge_asof(e.sort_values("_d"), a, left_on="_d", right_on="date", by="cik", direction="backward",
                          suffixes=("", "_att")).drop(columns=["_d", "date"])
        for c in ("instown", "coverage"):
            if c in e and e[c].notna().any():
                e[c] = (e[c] - e[c].mean()) / e[c].std()
    return e


def interactions(e: pd.DataFrame, y: str = "car_hold") -> dict:
    z = e["pct"] - 0.5
    out = {}
    for v, sign in PRED.items():
        if v not in e or e[v].isna().all():
            out[v] = {"available": False}
            continue
        d = pd.DataFrame({"y": e[y], "z": z, "v": e[v], "zv": z * e[v], "g": e["month"].astype(str)}).dropna()
        r = sm.OLS(d["y"], sm.add_constant(d[["z", "v", "zv"]])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["g"])[0]})
        t = float(r.tvalues["zv"])
        p_one = float(st.norm.sf(sign * t))
        out[v] = {"available": True, "coef": float(r.params["zv"]), "t": t, "predicted_sign": sign,
                  "p_one_sided": p_one, "pass": p_one < 0.05, "n": int(len(d))}
    return out
