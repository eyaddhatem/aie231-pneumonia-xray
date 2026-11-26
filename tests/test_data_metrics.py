import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import drop_exact_duplicates, make_splits, check_no_leakage, class_weights
from src.evaluate import binary_metrics
from src.train import log_run


def fake(n=1200, seed=0):
    r = np.random.default_rng(seed); rows = []
    for i in range(n):
        lab = int(r.random() < 0.73)
        rows.append(dict(path=f"/x/{i}.jpeg", orig_split="train", label_name="PNEUMONIA" if lab else "NORMAL", label=lab,
                         subtype="bacteria" if lab else "normal", patient_id=f"person{r.integers(0, 400)}" if lab else None,
                         md5=f"h{i if i % 40 else i - 1}"))
    return pd.DataFrame(rows)


def test_dedup_and_split():
    d = drop_exact_duplicates(fake()); assert d.md5.is_unique
    s = make_splits(d); check_no_leakage(s)
    assert abs((s.split == "train").mean() - 0.7) < 0.03
    assert (s[s.patient_id.notna()].groupby("patient_id").split.nunique() == 1).all()
    assert (make_splits(d).split.values == s.split.values).all()


def test_class_weights():
    w = class_weights(pd.DataFrame(dict(label=[0] * 10 + [1] * 30)))
    assert abs(w[0] - 2.0) < 1e-9 and abs(w[1] - 40 / 60) < 1e-9


def test_metrics_known_case():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1) and m["recall_pneumonia"] == 0.5 and m["auc"] == 0.75


def test_log_run_increments():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "a", "log.csv"); m = binary_metrics([0, 1], [0.2, 0.8])
        assert log_run(p, "me", "m", {}, 1, 1.0, 1, "val", m) == "R001"
        assert log_run(p, "me", "m", {}, 1, 1.0, 1, "val", m) == "R002"
