"""Laptop 1 entry point for the measured KAAVALU replay and dashboard."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from src.backend import check_authorization, get_auth_rules, init_db, save_detected_signals
from src.backend.db_manager import load_session
from src.rf_processing import classify_signals, extract_features, load_iq_recording
from src.rf_processing.dataset_generator import SCENARIOS, generate_simulated_disasternet_capture, simulate_recording
from src.rf_processing.iq_loader import analyze_recording


ROOT = Path(__file__).resolve().parent
RULE_FILE = ROOT / "config/disasternet_rules.json"
FAMILIAR_IQ = ROOT / "fixtures/familiar_clear.npz"


def _database():
    return os.environ.get("KAAVALU_DB_PATH", "kaavalu.db")


def _classify_input(source, conf_threshold=0.68, added_noise_db=0.0):
    if source == "fire_wide":
        # Held-out WB-FM seed. Only the simulator knows the waveform recipe.
        recording = simulate_recording("WB-FM", 61080, added_noise_db,
                                       tuning_hz=148225000.0, offset_hz=0.0)
        bursts, display = analyze_recording(recording)
    elif source in SCENARIOS and (source != "familiar_clear" or added_noise_db != 0):
        bursts, display = generate_simulated_disasternet_capture(source, added_noise_db)
    else:
        path = FAMILIAR_IQ if source == "familiar_clear" else Path(source)
        if added_noise_db != 0:
            raise ValueError("Noise adjustments require a generated scenario")
        if path.suffix.lower() != ".npz":
            raise ValueError("Choose a named simulation or a self-describing IQ .npz file")
        if not path.is_file():
            raise ValueError(f"IQ input does not exist: {path}")
        with np.load(path, allow_pickle=False) as header:
            rate = float(header["sample_rate_hz"].item())
            tuning = float(header["center_freq_hz"].item())
        recording = load_iq_recording(str(path), rate, tuning, fmt="npz")
        bursts, display = analyze_recording(recording)
    features = extract_features(bursts, display["sample_rate_hz"])
    classified = classify_signals(features, conf_threshold=conf_threshold)
    return classified, display


def run_kaavalu_pipeline(scenario_or_file, rules_list=None, conf_threshold=0.68,
                         added_noise_db=0.0, fmt="npz"):
    """Run RF, evaluate the current rules, and save detections and alerts together."""
    if fmt != "npz":
        raise ValueError("File input requires a self-describing NPZ with receiver metadata")
    started = perf_counter()
    path = _database()
    init_db(path, RULE_FILE)
    rules = get_auth_rules(path) if rules_list is None else rules_list
    classified, display = _classify_input(scenario_or_file, conf_threshold, added_noise_db)
    checked = check_authorization(classified, rules)
    save_detected_signals(checked, path)
    display["processing_latency_ms"] = (perf_counter() - started) * 1000
    return checked, display


def _dashboard_input(source, rules):
    # Laptop 3's callback consumer evaluates its selected rules and saves once.
    return _classify_input(source)


def _summary(row):
    fields = ("session_id", "signal_id", "source_tag", "sha256", "model_id", "center_freq_mhz",
              "snr_db", "occupied_bw_khz", "predicted_modulation", "peak_power_dbm",
              "calibration_status", "matched_rule_id", "verdict_category", "explainable_reason")
    return {key: row[key] for key in fields}


def _cli(args):
    db = _database()
    init_db(db, RULE_FILE)
    rules = get_auth_rules(db)
    checked, display = run_kaavalu_pipeline(args.source, rules)
    saved = {r["signal_id"]: r for r in load_session(display["capture_id"], db)}
    matched = {row["matched_rule_id"] for row in checked}
    result = {"input": args.source, "database": str(Path(db).resolve()),
              "source_tag": display["source_tag"], "processing_latency_ms": display["processing_latency_ms"],
              "matched_rules": [rule for rule in rules if rule["rule_id"] in matched],
              "records": [_summary(saved[r["signal_id"]]) for r in checked]}
    if args.rule_only:
        # The real row is already persisted. This copy is evaluated in memory only.
        if len(checked) != 1 or checked[0]["peak_power_dbm"] is not None:
            raise ValueError("The prepared demonstration requires one uncalibrated fire burst")
        prepared = dict(checked[0], signal_id=checked[0]["signal_id"] + "/rule-only",
                        peak_power_dbm=-20.0, calibration_status="HYPOTHETICAL_RULE_INPUT")
        example, = check_authorization([prepared], rules)
        result["rule_only_example"] = {"label": "PREPARED RULE-ONLY TEST",
                                       "hypothetical_power_dbm": -20.0,
                                       "derived_from_signal_id": checked[0]["signal_id"],
                                       "record": _summary(example)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    import sys

    if "--cli" in sys.argv or "--help" in sys.argv or "-h" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--cli", dest="source", nargs="?", const="familiar_clear", required=True,
                            metavar="INPUT", help="familiar_clear, fire_wide, another RF scenario, or an IQ .npz")
        parser.add_argument("--rule-only", action="store_true",
                            help="For fire_wide, also evaluate a labeled in-memory hypothetical power example")
        args = parser.parse_args()
        if args.rule_only and args.source != "fire_wide":
            parser.error("--rule-only requires --cli fire_wide")
        _cli(args)
    else:
        from src.dashboard import render_dashboard

        init_db(_database(), RULE_FILE)
        render_dashboard(_dashboard_input)
