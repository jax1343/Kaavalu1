"""Run with Python 3.11: python check_familiar_clear.py PATH_TO_EXTRACTED_REPLAY

Use --preview FILE.html to save the local dashboard preview.
"""

import argparse
from collections.abc import Mapping
from html import escape
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np

from src.contracts import evaluate_required_power, power_display_text, to_shared_record


def check_saved_replay(row, saved):
    assert row.keys() == saved.keys() and len(row) == 28, "RF field set changed"
    numeric_fields = {"features", "constellation_i", "constellation_q", "lof_normality_score"}
    for key in row:
        actual, expected = row[key], saved[key]
        if key not in numeric_fields:
            assert actual == expected, f"{key} differs from the saved replay"
        elif isinstance(expected, Mapping):
            assert actual.keys() == expected.keys(), f"{key} names changed"
            for name in expected:
                np.testing.assert_allclose(actual[name], expected[name], rtol=5e-4,
                                           atol=1e-12, err_msg=name)
        else:
            assert np.shape(actual) == np.shape(expected), f"{key} shape changed"
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, err_msg=key)


def unexpected_power_check(_power):
    raise AssertionError("a rule compared unavailable power")


def run(replay_dir):
    if sys.version_info[:2] != (3, 11) or np.__version__ != "1.26.4":
        raise RuntimeError("Use Python 3.11 and NumPy 1.26.4 for this replay")
    replay = subprocess.run([sys.executable, "-B", "replay_familiar_clear.py"],
                            cwd=replay_dir, capture_output=True, text=True, check=True)
    row = json.loads(replay.stdout)
    saved = json.loads((replay_dir / "handoff/familiar_clear_outgoing.json").read_text())
    check_saved_replay(row, saved)
    assert row["occupied_bw_hz"] == 16125 and "occupied_bw_khz" not in row
    shared = to_shared_record(row)
    assert shared["occupied_bw_khz"] == 16.125 and shared["peak_power_dbm"] is None
    assert "occupied_bw_hz" not in shared and to_shared_record(shared) == shared
    for key in row.keys() - {"occupied_bw_hz"}:
        assert row[key] == shared[key], f"adapter changed {key}"
    verdict = evaluate_required_power(shared["peak_power_dbm"], unexpected_power_check)
    assert verdict == "UNKNOWN"
    assert evaluate_required_power(0.0, lambda p: p == 0.0) is True

    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE TABLE trace (signal_id TEXT PRIMARY KEY, occupied_bw_khz REAL,"
                   " peak_power_dbm REAL, power_verdict TEXT, record_json TEXT)")
        db.execute("INSERT INTO trace VALUES (?, ?, ?, ?, ?)",
                   (row["signal_id"], shared["occupied_bw_khz"], shared["peak_power_dbm"],
                    verdict, json.dumps(shared, allow_nan=False)))
        db.commit()
        bandwidth, power, verdict, payload = db.execute(
            "SELECT occupied_bw_khz, peak_power_dbm, power_verdict, record_json "
            "FROM trace WHERE signal_id = ?", (row["signal_id"],)).fetchone()
    finally:
        db.close()
    stored = json.loads(payload)
    assert stored == shared and bandwidth == 16.125 and power is None and verdict == "UNKNOWN"
    labels = {
        "Signal ID": stored["signal_id"],
        "Source": stored["source_tag"],
        "Predicted modulation": stored["predicted_modulation"],
        "SNR": f'{stored["snr_db"]:.2f} dB',
        "Occupied bandwidth": f"{bandwidth} kHz",
        "Peak power": power_display_text(power),
        "Rule requiring power": f"{verdict} · Human review",
    }
    assert labels["Peak power"] == "Power unavailable."
    return labels


def save_preview(path, labels):
    rows = "".join(f"<tr><th scope='row'>{escape(k)}</th><td>{escape(v)}</td></tr>"
                   for k, v in labels.items())
    path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>familiar_clear replay</title><style>"
        "body{font:1rem/1.5 system-ui;max-width:40rem;margin:3rem auto;padding:0 1rem;"
        "color:#192e3a}table{border-collapse:collapse;width:100%}"
        "td,th{text-align:left;padding:.6rem;border-bottom:1px solid #ccd6db}"
        "</style><h1>familiar_clear</h1><p>Local dashboard preview from the SQLite replay test.</p>"
        f"<table>{rows}</table></html>", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Run without -O so the replay assertions are enabled")
    labels = run(args.replay_dir)
    if args.preview:
        save_preview(args.preview, labels)
    print(json.dumps(labels, indent=2))
