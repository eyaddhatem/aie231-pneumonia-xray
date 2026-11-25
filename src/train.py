"""Experiment logging (spec 3.5). The training loop is added in Milestone 4."""
import csv
import datetime as dt
import os

LOG_COLUMNS = ["run_id", "date", "author", "model", "hyperparams", "seed", "train_time_s",
               "params", "split", "accuracy", "recall_pneumonia", "recall_normal",
               "precision_pneumonia", "f1_pneumonia", "macro_f1", "auc", "notes"]


def log_run(path: str, author: str, model: str, hyperparams: dict, seed: int, train_time_s: float,
            params, split: str, metrics: dict, notes: str = "") -> str:
    """Append one row to the experiment CSV (creates it with a header) and return the run id."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    n = 1 if new else sum(1 for _ in open(path)) 
    run_id = f"R{n:03d}"
    row = dict(run_id=run_id, date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), author=author, model=model,
               hyperparams=str(hyperparams), seed=seed, train_time_s=round(train_time_s, 1), params=params,
               split=split, notes=notes, **{k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in metrics.items() if k in LOG_COLUMNS})
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return run_id
