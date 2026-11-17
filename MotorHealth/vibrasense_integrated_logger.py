#!/usr/bin/env python3
"""Robust VibraSense logger — does not wait for HEADER."""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
from datetime import datetime
try:
    import serial
except ImportError:
    print("Install pyserial: py -m pip install pyserial")
    raise SystemExit(1)

COLUMNS = [
    "session_id", "window_id", "elapsed_s", "fan_state", "speed_setting", "mounting",
    "accel_samples", "accel_sample_rate_hz",
    "ax_rms_g", "ay_rms_g", "az_rms_g",
    "ax_peak_g", "ay_peak_g", "az_peak_g", "fft_peak_hz",
    "audio_samples", "audio_sample_rate_hz",
    "audio_rms_raw", "audio_peak_raw", "audio_zcr",
    "rpm_pulses", "rpm_est", "ds18b20_c", "temperature_valid",
    "i2c_errors", "fifo_overruns"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session", default="baseline")
    ap.add_argument("--fan-state", default="healthy")
    ap.add_argument("--speed-setting", default="1")
    ap.add_argument("--mounting", default="housing")
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--out", default="vibrasense_data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.session}_{stamp}.csv"

    try:
        with serial.Serial(args.port, args.baud, timeout=0.5) as ser, \
             csv_path.open("w", newline="", encoding="utf-8") as csv_f:
            print(f"Connected to {args.port}. Recording {args.duration}s...")
            ser.reset_input_buffer()
            writer = csv.DictWriter(csv_f, fieldnames=COLUMNS)
            writer.writeheader()

            start = time.monotonic()
            rows = 0
            while (time.monotonic() - start) < args.duration:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line.startswith("DATA,"):
                    continue
                vals = next(csv.reader([line]))[1:]
                if len(vals) != len(COLUMNS):
                    print(f"\n⚠ Expected {len(COLUMNS)} cols, got {len(vals)} — skipping")
                    continue
                row = dict(zip(COLUMNS, vals))
                row["fan_state"] = args.fan_state
                row["speed_setting"] = args.speed_setting
                row["mounting"] = args.mounting
                writer.writerow(row)
                rows += 1
                if rows % 5 == 0:
                    print(".", end="", flush=True)
            print(f"\n✅ Saved {rows} windows → {csv_path}")
            return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())