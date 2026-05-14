import argparse
import json
import math
from typing import Any, List, Dict


def _mean_lists(values: List[Any]) -> Any:
    """
    Recursively compute elementwise mean of nested list structures.
    Supports numbers and (nested) lists of numbers.
    """
    if not values:
        return values

    first = values[0]
    if isinstance(first, (int, float)):
        s = 0.0
        for v in values:
            s += float(v)
        return s / float(len(values))

    if isinstance(first, list):
        ln = len(first)
        out = []
        for i in range(ln):
            out.append(_mean_lists([v[i] for v in values]))
        return out

    raise TypeError(f"Unsupported value type for mean: {type(first)}")


def aggregate(updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights_list = []
    losses = []
    for u in updates:
        w = u.get("weights") or {}
        weights_list.append(w)
        if "loss" in u and isinstance(u["loss"], (int, float)) and math.isfinite(float(u["loss"])):
            losses.append(float(u["loss"]))

    if not weights_list:
        return {"weights": {}, "avg_loss": None}

    keys = list(weights_list[0].keys())
    out_weights = {}
    for k in keys:
        vals = [w[k] for w in weights_list if k in w]
        if not vals:
            continue
        out_weights[k] = _mean_lists(vals)

    avg_loss = sum(losses) / len(losses) if losses else None
    return {"weights": out_weights, "avg_loss": avg_loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    updates = payload.get("updates", [])
    result = aggregate(updates)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

