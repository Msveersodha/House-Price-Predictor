"""Inspect a saved scikit-learn model WITHOUT guessing anything.

Usage:
    python inspect_model.py app/models/house_price_model.pkl --trust
    python inspect_model.py app/models/house_price_model.pkl --trust --sample '{"MedInc": 8.3, ...}'

SECURITY: .pkl files can run arbitrary code when loaded. Only inspect files you
created yourself or fully trust. Because of that, --trust is mandatory.
Never load a pickle that came from someone you don't trust.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path


def describe_estimator(obj, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{pad}- {type(obj).__module__}.{type(obj).__name__}")
    steps = getattr(obj, "steps", None)  # Pipeline
    if steps:
        for name, step in steps:
            print(f"{pad}  step '{name}':")
            describe_estimator(step, indent + 4)
    transformers = getattr(obj, "transformers_", None)  # ColumnTransformer
    if transformers:
        for name, tr, cols in transformers:
            print(f"{pad}  transformer '{name}' on columns {cols}:")
            if tr not in ("drop", "passthrough"):
                describe_estimator(tr, indent + 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--trust", action="store_true", help="confirm you trust this file")
    ap.add_argument("--sample", help="JSON object with one row of feature values to run a test prediction")
    args = ap.parse_args()

    if not args.path.is_file():
        print(f"File not found: {args.path}")
        return 1
    if not args.trust:
        print("Refusing to load: pass --trust only if this is your own, trusted model file.")
        return 2

    raw = args.path.read_bytes()
    print(f"File: {args.path}  ({len(raw):,} bytes)")
    print(f"SHA-256: {hashlib.sha256(raw).hexdigest()}")

    import joblib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(args.path)
        except ModuleNotFoundError as exc:
            print(f"\nCannot load: missing package '{exc.name}'. pip install -r requirements.txt")
            return 1
    for w in caught:
        print(f"WARNING: {w.message}")

    import sklearn

    print(f"\nInstalled scikit-learn: {sklearn.__version__}")
    print(f"\nObject type: {type(model).__module__}.{type(model).__name__}")
    is_pipeline = hasattr(model, "steps")
    print("Is a scikit-learn Pipeline:", is_pipeline)
    print("Structure:")
    describe_estimator(model)

    final = model.steps[-1][1] if is_pipeline else model
    print(f"\nFinal estimator: {type(final).__name__}  (has predict: {hasattr(final, 'predict')})")
    for key in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "random_state"):
        if hasattr(final, key):
            print(f"  {key} = {getattr(final, key)}")

    names = getattr(model, "feature_names_in_", None)
    n_in = getattr(model, "n_features_in_", None)
    print("\nExpected input dimensions:", f"(n_samples, {n_in})" if n_in is not None else "unknown")
    if names is not None:
        print("Feature names (in required order):")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")
    else:
        print(
            "Feature names NOT stored in the model.\n"
            "You must get them (and their order) from the training notebook / dataset."
        )

    first = model.steps[0][1] if is_pipeline else None
    if first is not None and hasattr(first, "statistics_"):
        print(f"\nPreprocessing step '{type(first).__name__}': strategy={getattr(first, 'strategy', '?')}")
        cols = getattr(first, "feature_names_in_", range(len(first.statistics_)))
        for c, v in zip(cols, first.statistics_):
            print(f"  fill value for {c}: {v:.4f}")

    if args.sample:
        import pandas as pd

        row = json.loads(args.sample)
        frame = pd.DataFrame([row], columns=list(names) if names is not None else None)
        print("\nTest prediction:", model.predict(frame))
    return 0


if __name__ == "__main__":
    sys.exit(main())
