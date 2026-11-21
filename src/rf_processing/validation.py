"""Run reproducible RF-to-policy scenarios and capture-level evaluation offline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from src.backend import check_authorization, get_auth_rules, init_db, save_detected_signals
from src.backend.db_manager import load_session
from .classifier import MODEL_ID, classify_signals, train_reference_model
from .dataset_generator import CLASSES, FS, TUNING, simulate_recording
from .experiment import held_out_trials
from .feature_extractor import extract_features
from .iq_loader import analyze_recording, load_iq_recording


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "familiar_clear.npz"
FIELDS = ("signal_id", "source_tag", "sha256", "sample_count", "center_freq_hz",
          "occupied_bw_khz", "snr_db", "peak_power_dbm", "calibration_status",
          "model_id", "candidate_modulation", "predicted_modulation", "classification_confidence",
          "lof_normality_score", "clf_reason", "matched_rule_id", "verdict_category", "explainable_reason")


def _scenario_inputs():
    yield "familiar_clear", {"file": "fixtures/familiar_clear.npz"}, ("QPSK", None, "RECOGNIZED_NO_AUTH_RECORD")
    yield "authorized_qpsk", dict(family="QPSK", seed=81001, tuning_hz=148532000, offset_hz=18000), (
        "QPSK", "EM-RES-04", "KNOWN_AUTHORIZED")
    yield "fire_wide", dict(family="WB-FM", seed=61080, tuning_hz=148225000, offset_hz=0), (
        "WB-FM", "EM-FIR-02", "UNKNOWN")
    yield "unfamiliar_clear", dict(family="16-QAM", seed=91001, offset_hz=18000), (
        "UNKNOWN", None, "UNKNOWN")
    yield "weak_qpsk", dict(family="QPSK", seed=81001, offset_hz=18000, added_noise_db=16), (
        "UNKNOWN", None, "UNKNOWN")
    yield "missed_qpsk", dict(family="QPSK", seed=81001, offset_hz=18000, added_noise_db=18), None


def _outcome(rows, expected):
    keys = ("predicted_modulation", "matched_rule_id", "verdict_category")
    actual = [tuple(row[k] for k in keys) for row in rows]
    return {"expected": dict(zip(keys, expected)) if expected else {"detections": 0},
            "passed": actual == ([expected] if expected else []),
            "records": [{k: row[k] for k in FIELDS} for row in rows]}


def run_scenarios(model, db, rules):
    """Caller supplies a fresh validation DB. Hypothetical rows stay in memory."""
    scenarios = {}
    for name, inputs, expected in _scenario_inputs():
        start = perf_counter()
        if "file" in inputs:
            recording = load_iq_recording(str(FIXTURE), FS, TUNING, fmt="npz")
        else:
            recording = simulate_recording(**inputs)
        bursts, meta = analyze_recording(recording)
        native = classify_signals(extract_features(bursts, meta["sample_rate_hz"]), model)
        policy = rules
        override = None
        if name == "authorized_qpsk":
            override = dict(rule_id="EM-RES-04", max_power_dbm=None)
            policy = [dict(r, **override) if r["rule_id"] == override["rule_id"] else r for r in rules]
        checked = check_authorization(native, policy)
        save_detected_signals(checked, db)
        stored = load_session(meta["capture_id"], db)
        result = _outcome(stored, expected)
        result.update(inputs=inputs, capture_id=meta["capture_id"], sha256=meta["sha256"],
                      policy_override=override, source_tag=meta["source_tag"],
                      sqlite_round_trip=(stored == checked),
                      latency_ms=(perf_counter() - start) * 1000)
        result["passed"] &= result["sqlite_round_trip"]
        scenarios[name] = result
        if name == "fire_wide":
            supplied = {"peak_power_dbm": -20.0, "calibration_status": "HYPOTHETICAL_RULE_INPUT"}
            hypothetical = [dict(row, **supplied, signal_id=row["signal_id"] + "/rule-only") for row in native]
            tested = check_authorization(hypothetical, rules)
            example = _outcome(tested, ("WB-FM", "EM-FIR-02", "POTENTIALLY_UNAUTHORIZED"))
            example.update(inputs=supplied, parent_capture=meta["capture_id"], kind="rule-only, in memory")
            scenarios["fire_rule_only"] = example
    return scenarios


def _rate(n, d):
    return n / d if d else None


def score_trials(trials):
    """Score every parent once. Abstentions, misses and multiple outputs remain in denominators."""
    labels = (*CLASSES, "UNKNOWN")
    confusion = {label: Counter() for label in labels}
    for trial in trials:
        truth = "UNKNOWN" if trial["family"] == "16-QAM" else trial["family"]
        confusion[truth][trial["prediction"]] += 1
    metrics = {}
    for label, row in confusion.items():
        support = sum(row.values())
        tp = row[label]
        fp = sum(other[label] for name, other in confusion.items() if name != label)
        metrics[label] = dict(support=support, tp=tp, fp=fp, fn=support-tp,
                              precision=_rate(tp, tp+fp), recall=_rate(tp, support),
                              false_positive_rate=_rate(fp, len(trials)-support))
    known_n = sum(metrics[label]["support"] for label in CLASSES)
    correct_known = sum(metrics[label]["tp"] for label in CLASSES)
    unknown = confusion["UNKNOWN"]
    return dict(confusion=confusion, per_class=metrics,
                known_accuracy=_rate(correct_known, known_n),
                overall_accuracy=_rate(correct_known + unknown["UNKNOWN"], len(trials)),
                unfamiliar_accepted=sum(unknown[label] for label in CLASSES),
                known_sent_to_review=sum(confusion[label]["UNKNOWN"] for label in CLASSES))


def evaluate(rules_path=None):
    start = perf_counter()
    model = train_reference_model()
    fit_ms = (perf_counter() - start) * 1000
    with TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "scenarios.db")
        init_db(db, rules_path)
        rules = get_auth_rules(db)
        scenarios = run_scenarios(model, db, rules)
    # These seeds are separate from the published experiment and reference training.
    trials = list(held_out_trials(model, known_start=73000, unfamiliar_start=96000))
    times = [row["latency_ms"] for row in trials]
    return dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                processor=platform.processor(), model_id=MODEL_ID, reference_fit_ms=fit_ms,
                rules=rules, rules_sha256=hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest(),
                scenarios=scenarios, scores=score_trials(trials), trials=trials,
                rf_latency_ms=dict(n=len(times), median=float(np.median(times)), p95=float(np.percentile(times,95))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, help="DisasterNet rule JSON for a fresh validation database")
    parser.add_argument("--output", type=Path, help="Save complete JSON evidence")
    args = parser.parse_args()
    evidence = evaluate(args.rules)
    payload = json.dumps(evidence, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    raise SystemExit(0 if all(s["passed"] for s in evidence["scenarios"].values()) else 1)
