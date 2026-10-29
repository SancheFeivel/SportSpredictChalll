"""Append one record per training run to tracking/weights_history.jsonl.
Call from train_all.py after the loop:  track.log_run(saved, rows, n_games=len(df))
Stored per question: coefs (scaled space), intercept, Brier vs baseline, edge, coef drift vs previous run.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

PATH = "tracking/weights_history.jsonl"


def _coefs(spec):
    if spec.get("type") != "model":
        return None
    lr = spec["model"][-1]  # LogisticRegression at end of pipeline
    return dict(zip(spec["features"], map(float, lr.coef_[0]))), float(lr.intercept_[0])


def load():
    if not os.path.exists(PATH):
        return []
    with open(PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def log_run(saved, rows, n_games, note=""):
    os.makedirs("tracking", exist_ok=True)
    hist = load()
    prev = hist[-1]["questions"] if hist else {}
    stats = {r["question"]: r for r in rows}
    qs = {}
    for q, spec in saved.items():
        r = stats[q]
        rec = {
            "use": r["use"], "base_brier": r["base_brier"],
            "model_brier": r["model_brier"], "edge_pts": r["edge_pts"],
            "const_p": spec["p"], "coef": None, "intercept": None, "drift": None,
        }
        c = _coefs(spec)
        if c:
            rec["coef"], rec["intercept"] = c
            old = (prev.get(q) or {}).get("coef")
            if old:
                keys = set(old) | set(rec["coef"])
                d = np.array([rec["coef"].get(k, 0) - old.get(k, 0) for k in keys])
                rec["drift"] = float(np.linalg.norm(d))  # L2 change in weights
        qs[q] = rec
    entry = {
        "run": len(hist) + 1,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_games": int(n_games), "note": note, "questions": qs,
    }
    with open(PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"tracked run {entry['run']} -> {PATH}")
