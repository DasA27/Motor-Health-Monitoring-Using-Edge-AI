#!/usr/bin/env python3
"""VibraSense Multimodal Trainer (13 features)."""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "ax_rms_g", "ay_rms_g", "az_rms_g",
    "ax_peak_g", "ay_peak_g", "az_peak_g",
    "fft_peak_hz",
    "audio_rms_raw", "audio_peak_raw", "audio_zcr",
    "rpm_est", "ds18b20_c", "fifo_overruns"
]

def expand_inputs(items):
    out = []
    for item in items:
        out.extend([Path(p) for p in glob.glob(str(item))])
    return list(dict.fromkeys(out))

def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [Skipping] {path.name}: {e}")
        return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser(description="VibraSense Trainer")
    ap.add_argument("training_csv", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, default=Path("models/vibrasense_model.joblib"))
    ap.add_argument("--healthy-value", default="healthy")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--export-edge-header", type=Path)
    args = ap.parse_args()

    try:
        paths = expand_inputs(args.training_csv)
        frames = [read_csv(p) for p in paths]
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise ValueError("No valid data found.")

        df = pd.concat(frames, ignore_index=True)
        healthy = df[df["fan_state"].astype(str).str.lower() == args.healthy_value.lower()].copy()
        if healthy.empty:
            raise ValueError("No rows with fan_state='healthy'. Check CSVs.")

        for col in FEATURES:
            if col not in healthy.columns:
                healthy[col] = 0.0
            healthy[col] = pd.to_numeric(healthy[col], errors="coerce")
        healthy["ds18b20_c"] = healthy["ds18b20_c"].fillna(25.0)
        healthy["rpm_est"] = healthy["rpm_est"].fillna(0.0)
        healthy[FEATURES] = healthy[FEATURES].fillna(0.0)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(n_estimators=300, contamination=args.contamination, random_state=42, n_jobs=-1)),
        ])
        X = healthy[FEATURES].astype(float)
        pipeline.fit(X)

        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "features": FEATURES}, args.model)

        if args.export_edge_header:
            medians = X.median(axis=0).to_numpy(dtype=float)
            mads = (X - medians).abs().median(axis=0).to_numpy(dtype=float)
            scales = 1.4826 * mads
            for i, col in enumerate(FEATURES):
                if col in ["ax_rms_g","ay_rms_g","az_rms_g","ax_peak_g","ay_peak_g","az_peak_g"]:
                    scales[i] = max(scales[i], abs(medians[i]) * 0.5, 0.1)
                elif col == "fft_peak_hz":
                    scales[i] = max(scales[i], 10.0)
                elif col in ["audio_rms_raw","audio_peak_raw","audio_zcr"]:
                    scales[i] = max(scales[i], abs(medians[i]) * 0.5, 1000.0)
                elif col == "rpm_est":
                    scales[i] = max(scales[i], 50.0)
                else:
                    scales[i] = max(scales[i], 1.0)

            def cf(v):
                t = f"{v:.9g}"
                if "." not in t and "e" not in t.lower(): t += ".0"
                return t + "f"

            lines = [
                "#pragma once",
                "#define VIBRASENSE_EDGE_MODEL_READY 1",
                f"#define VIBRASENSE_EDGE_FEATURE_COUNT {len(FEATURES)}",
                ("static const float VIBRASENSE_EDGE_MEDIAN[VIBRASENSE_EDGE_FEATURE_COUNT] = {"
                 + ", ".join(cf(v) for v in medians) + "};"),
                ("static const float VIBRASENSE_EDGE_SCALE[VIBRASENSE_EDGE_FEATURE_COUNT] = {"
                 + ", ".join(cf(v) for v in scales) + "};"),
                "",
            ]
            args.export_edge_header.parent.mkdir(parents=True, exist_ok=True)
            args.export_edge_header.write_text("\n".join(lines), encoding="utf-8")

        print(f"\n✅ Trained!")
        print(f"Usable windows:  {len(healthy)}")
        print(f"Features:        {len(FEATURES)}")
        if args.export_edge_header:
            print(f"Header saved:    {args.export_edge_header.resolve()}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())