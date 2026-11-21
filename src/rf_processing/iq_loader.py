"""Headerless little-endian I/Q pairs or a self-describing simulation archive."""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, is_zipfile

import numpy as np


def check_receiver(sample_rate_hz, center_freq_hz):
    for name, value in (("sample_rate_hz", sample_rate_hz), ("center_freq_hz", center_freq_hz)):
        if not np.isscalar(value) or not isinstance(value, (int, float, np.number)) or not np.isrealobj(value):
            raise ValueError(f"{name} must be a finite number in Hz")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite in Hz")


def load_iq_recording(file_path: str, sample_rate_hz: float, center_freq_hz: float,
                      fmt: str = "complex64", source_tag: str = "RECORDED_IQ") -> dict:
    """Raw replay needs acquisition metadata. NPZ embeds source, tuning and rate."""
    path = Path(file_path)
    check_receiver(sample_rate_hz, center_freq_hz)
    if source_tag not in {"RECORDED_IQ", "SIMULATED"}:
        raise ValueError("File replay accepts RECORDED_IQ or SIMULATED as its source_tag")
    if not path.is_file():
        raise ValueError(f"IQ file does not exist: {path}")
    if fmt == "npz":
        try:
            with np.load(path, allow_pickle=False) as data:
                iq = data["iq_data"]
                origin = str(data["source_tag"].item())
                if origin not in {"RECORDED_IQ", "SIMULATED"}:
                    raise ValueError("Invalid replay provenance")
                if data["sample_rate_hz"].item() != sample_rate_hz or data["center_freq_hz"].item() != center_freq_hz:
                    raise ValueError("Receiver metadata disagrees with saved capture")
                if iq.dtype != np.complex64:
                    raise ValueError("Expected complex64 archive samples")
                if origin == "SIMULATED":
                    source_tag = origin
        except (OSError, KeyError, EOFError, TypeError, BadZipFile) as exc:
            raise ValueError(f"Malformed IQ archive: {exc}") from exc
    elif fmt in {"complex64", "interleaved_float32", "float32_interleaved"}:
        with path.open("rb") as stream:
            if stream.read(4) == b"PK\x03\x04" and is_zipfile(path):
                raise ValueError("ZIP/NPZ header found; use fmt='npz' for a saved capture")
        if path.stat().st_size == 0 or path.stat().st_size % 8:
            raise ValueError("Expected a nonempty file with complete 8-byte I/Q pairs")
        iq = np.memmap(path, mode="r", dtype="<c8")
    else:
        raise ValueError(f"Unsupported IQ format: {fmt}")
    if iq.ndim != 1 or iq.size == 0 or not np.isfinite(iq).all():
        raise ValueError("Expected a nonempty finite one-dimensional IQ recording")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(iq_data=(iq - np.complex64(iq.mean(dtype=np.complex128))).astype(np.complex64, copy=False),
                source_tag=source_tag, sample_rate_hz=float(sample_rate_hz),
                center_freq_hz=float(center_freq_hz), sample_count=len(iq), sha256=digest)


def analyze_recording(recording, min_snr_db=6.0):
    """Join caller-supplied provenance to the detector's unlabelled measurements."""
    from .signal_detector import detect_signals

    source = recording.get("source_tag")
    if source not in {"SIMULATED", "RECORDED_IQ", "LIVE_SDR"}:
        raise ValueError("Missing or invalid acquisition source_tag")
    signals, display = detect_signals(recording["iq_data"], recording["sample_rate_hz"],
                                      recording["center_freq_hz"], min_snr_db)
    digest = recording.get("sha256") or hashlib.sha256(np.ascontiguousarray(recording["iq_data"])).hexdigest()
    acquisition = [digest, recording["sample_rate_hz"], recording["center_freq_hz"], source]
    capture_id = hashlib.sha256(json.dumps(acquisition).encode()).hexdigest()[:20]
    provenance = dict(source_tag=source, capture_id=capture_id, sha256=digest)
    display.update(provenance)
    for row in signals:
        row.update(provenance)
        row["signal_id"] = f"{capture_id}/{row['signal_id']}"
    return signals, display
