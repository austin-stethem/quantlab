"""No-look-ahead and plumbing tests for quantlab v3. Run: python -m pytest -q tests"""
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from v3 import core as K, synth, trust as T


@pytest.fixture(scope="module")
def small():
    return synth.make_market(seed=3, start="2012-01-02", end="2026-11-13", n_members=120, n_extra=40)


def test_day0_rules():
    days = pd.bdate_range("2024-03-04", "2024-03-15")   # Mon 4 .. Fri 15
    acc = pd.Series(pd.to_datetime(["2024-03-04 07:00", "2024-03-04 12:00", "2024-03-04 16:05",
                                    "2024-03-08 16:30", "2024-03-09 09:00"]))
    got = days[K.assign_day0(acc, days)]
    want = pd.to_datetime(["2024-03-04", "2024-03-04", "2024-03-05", "2024-03-11", "2024-03-11"])
    assert list(got) == list(want)


def test_dedupe_keeps_the_real_release():
    fil = pd.DataFrame({"cik": 1, "form": ["8-K", "8-K", "8-K", "8-K/A"],
                        "accepted": pd.to_datetime(["2024-02-01 09:00", "2024-04-10 08:00", "2024-04-25 16:10", "2024-04-28 10:00"])})
    per = pd.DataFrame({"cik": [1], "period_end": pd.to_datetime(["2024-03-31"]), "filed": pd.to_datetime(["2024-05-05"])})
    ev = K.dedupe_events(fil, per)
    assert len(ev) == 1 and ev["accepted"].iloc[0] == pd.Timestamp("2024-04-25 16:10")


def test_deciles_never_see_the_future(small):
    D = K.dev_slice(small)
    ev = K.compute_ear(D, K.dedupe_events(D.filings, D.periods))
    full = K.assign_deciles(ev).set_index(["cik", "accepted"])["pct"]
    cut = ev["signal_idx"].quantile(0.5)
    early = K.assign_deciles(ev[ev["signal_idx"] <= cut]).set_index(["cik", "accepted"])["pct"]
    pd.testing.assert_series_equal(full.loc[early.index], early, check_names=False)


def test_calendar_starts_after_entry_and_charges_cost(small):
    D = K.dev_slice(small)
    c = int(np.where(np.isfinite(D.R[:, 0]))[0][0])
    ev = pd.DataFrame({"entry_idx": [300], "col": [0], "decile": [10]})
    cal = K.calendar(D, ev, H=5, cost=0.001)
    r = D.R[301:306, 0].copy()
    r[0] -= 0.001; r[-1] -= 0.001
    assert np.allclose(cal[10].values[301:306], r)
    assert cal[10].values[300] == 0 and cal[10].values[306] == 0


def test_holdout_is_invisible(small):
    D = K.dev_slice(small)
    assert D.days.max() <= K.DEV_END
    assert D.filings["accepted"].max() < K.HOLDOUT_START
    assert D.periods["filed"].max() <= K.DEV_END
    assert D.ff.index.max() <= K.DEV_END


def test_event_schema(small):
    D = K.dev_slice(small)
    ev = K.assign_deciles(K.compute_ear(D, K.dedupe_events(D.filings, D.periods)))
    schema = pa.DataFrameSchema({
        "pct": pa.Column(float, pa.Check.in_range(0, 1)),
        "decile": pa.Column(int, pa.Check.in_range(1, 10)),
        "ear": pa.Column(float, pa.Check(np.isfinite)),
    }, checks=[pa.Check(lambda d: (d["signal_idx"] == d["day0"] + 1).all()),
               pa.Check(lambda d: (d["entry_idx"] == d["day0"] + K.ENTRY_LAG).all())])
    schema.validate(ev)


def test_dating_estimator_matches_truth(small):
    D = K.dev_slice(small)
    ew = K.ew_universe(D)
    ev = K.compute_ear(D, K.dedupe_events(D.filings, D.periods), ew)
    j = ev.merge(small.meta["truth"], on=["cik", "accepted"])
    truth = (j["day0"] == j["true_d0"]).mean()
    assert abs(T.dating_quality(D, ev, ew)["correct_share"] - truth) < 0.03


def test_planted_drift_is_recovered(small):
    D = K.dev_slice(small)
    ew = K.ew_universe(D)
    ev = K.assign_deciles(K.compute_ear(D, K.dedupe_events(D.filings, D.periods), ew))
    r = T.recovery(D, ev, ew, delta=0.03)
    assert 0.75 <= r["ratio"] <= 1.25
