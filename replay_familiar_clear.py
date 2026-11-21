"""Replay the bundled capture from disk, printing one untouched outgoing dictionary as JSON."""
import json
from pathlib import Path

from src.rf_processing import classify_signals, extract_features, load_iq_recording
from src.rf_processing.iq_loader import analyze_recording


def replay():
    path = Path(__file__).resolve().parent / "fixtures" / "familiar_clear.npz"
    recording = load_iq_recording(str(path), 192_000., 150_000_000., fmt="npz")
    signals, display = analyze_recording(recording)
    records = classify_signals(extract_features(signals, recording["sample_rate_hz"]))
    row, = records
    return row, display


if __name__ == "__main__":
    result, _ = replay()
    print(json.dumps(result, indent=2, allow_nan=False))
