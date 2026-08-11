"""
benchmarking.py
----------------
Clinical-style agreement statistics (same methodology as the notebook's
evaluate.py), reused by the web app to benchmark predicted differential
counts against user-supplied reference (ground-truth) counts.
"""

from collections import defaultdict

import numpy as np


def bland_altman_stats(reference, predicted):
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    diffs = predicted - reference
    means = (predicted + reference) / 2
    bias = float(diffs.mean()) if len(diffs) else 0.0
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "means": means.tolist(),
        "diffs": diffs.tolist(),
    }


def pearson_r(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def counts_by_group(records, key_field, label_field, classes):
    """records: list of dicts with at least `key_field` and `label_field`.
    Returns {group: {class: count}}."""
    counts = defaultdict(lambda: {c: 0 for c in classes})
    for r in records:
        group = r[key_field]
        label = r[label_field]
        if label in counts[group]:
            counts[group][label] += 1
        else:
            counts[group][label] = 1
    return counts


def benchmark_predictions(predicted_records, reference_records, classes):
    """
    predicted_records / reference_records: list of {"group": ..., "label": ...}
    Returns per-class Bland-Altman + correlation stats comparing predicted
    vs reference counts, aggregated per group.
    """
    pred_counts = counts_by_group(predicted_records, "group", "label", classes)
    ref_counts = counts_by_group(reference_records, "group", "label", classes)

    groups = sorted(set(pred_counts.keys()) | set(ref_counts.keys()))

    out = {}
    for cls in classes:
        pred = [pred_counts.get(g, {}).get(cls, 0) for g in groups]
        ref = [ref_counts.get(g, {}).get(cls, 0) for g in groups]
        stats = bland_altman_stats(ref, pred)
        out[cls] = {
            "groups": groups,
            "reference_counts": ref,
            "predicted_counts": pred,
            "pearson_r": pearson_r(ref, pred),
            **stats,
        }
    return out
