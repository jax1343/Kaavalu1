"""Offline replay of a bounded, headerless little-endian float32 I/Q file."""
import argparse
import json
from pathlib import Path

from .classifier import classify_signals
from .feature_extractor import extract_features
from .iq_loader import analyze_recording, load_iq_recording


def replay_raw(path, sample_rate_hz, center_freq_hz, fmt, source_tag):
    if fmt not in {"complex64", "interleaved_float32"}:
        raise ValueError("Expected complex64 or interleaved_float32")
    if source_tag not in {"SIMULATED", "RECORDED_IQ"}:
        raise ValueError("Declare the file origin as SIMULATED or RECORDED_IQ")
    # The existing detector holds the spectrogram in RAM. Limit this entry
    # point to one million complete I/Q pairs until streaming is validated.
    if Path(path).stat().st_size > 8_000_000:
        raise ValueError("Raw replay supports up to 1,000,000 complex samples")
    recording = load_iq_recording(str(path), sample_rate_hz, center_freq_hz, fmt, source_tag)
    bursts, display = analyze_recording(recording)
    rows = classify_signals(extract_features(bursts, sample_rate_hz))
    return {"capture_id": display["capture_id"], "sha256": recording["sha256"],
            "source_tag": recording["source_tag"], "sample_count": recording["sample_count"],
            "sample_rate_hz": sample_rate_hz, "tuning_freq_hz": center_freq_hz, "records": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument("--center-freq-hz", type=float, required=True)
    parser.add_argument("--format", choices=["complex64", "interleaved_float32"], required=True)
    parser.add_argument("--source", choices=["SIMULATED", "RECORDED_IQ"], required=True)
    args = parser.parse_args()
    print(json.dumps(replay_raw(args.path, args.sample_rate_hz, args.center_freq_hz,
                                args.format, args.source), indent=2, allow_nan=False))
