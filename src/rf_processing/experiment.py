"""Run offline from the checkout root: python -m src.rf_processing.experiment"""
import argparse
from collections import Counter
import json

import numpy as np

from .classifier import classify_signals, train_reference_model
from .dataset_generator import CLASSES, FS, TUNING, generate_simulated_disasternet_capture, simulate_recording
from .feature_extractor import extract_features
from .iq_loader import analyze_recording


def _classify(signals, model):
    return classify_signals(extract_features(signals, FS), model)


def _brief(row):
    return {key: row[key] for key in ("signal_id", "source_tag", "center_freq_hz", "occupied_bw_hz",
            "snr_db", "candidate_modulation", "predicted_modulation", "classification_confidence",
            "lof_normality_score", "clf_reason", "peak_power_dbm")}


def experiment(rescue_frequency_hz=None):
    model = train_reference_model()
    report = {"data_source": "SIMULATED", "samples_per_capture": 12288,
              "training_counts": dict(Counter(model["labels"])), "examples": {}}
    for scenario in ("familiar_clear", "familiar_weak", "unfamiliar_clear"):
        signals, display = generate_simulated_disasternet_capture(scenario)
        report["examples"][scenario] = {"added_noise_db": display.get("added_noise_db", 0.),
                                        "detections": [_brief(r) for r in _classify(signals, model)]}
    # Same QPSK data, phase and standardized noise. Only the noise scale changes.
    for addition in np.arange(0., 41., .5):
        rec = simulate_recording("QPSK", 81001, addition, offset_hz=18000.)
        signals, _ = analyze_recording(rec)
        if not signals:
            report["first_missed_qpsk_added_noise_db"] = float(addition)
            break
    target = TUNING + 18000 if rescue_frequency_hz is None else rescue_frequency_hz
    rescue = simulate_recording("FM-NB", 82001, tuning_hz=target-18000., offset_hz=18000.)
    signals, _ = analyze_recording(rescue)
    report["rescue_rule_status"] = ("Illustrative tuning; team corridor file unavailable"
                                    if rescue_frequency_hz is None else "Operator supplied simulated corridor tuning")
    report["examples"]["rescue_corridor"] = [_brief(r) for r in _classify(signals, model)]
    # Independent fixed test seeds; no 16-QAM is used by fitting or threshold selection.
    report["known_test"] = {}
    for c, name in enumerate(CLASSES):
        counts = Counter()
        for j in range(16):
            capture = simulate_recording(name, 61000 + 16*c + j, added_noise_db=2*(j%5))
            bursts, _ = analyze_recording(capture)
            labels = [r["predicted_modulation"] for r in _classify(bursts, model)]
            counts[labels[0] if len(labels)==1 else ("MISSED" if not labels else "MULTIPLE")] += 1
        report["known_test"][name] = dict(counts)
    report["held_out_16qam"] = []
    for j in range(12):
        capture = simulate_recording("16-QAM", 92000+j, added_noise_db=j%4)
        bursts, _ = analyze_recording(capture)
        report["held_out_16qam"].append({"seed": 92000+j, "detections": [_brief(r) for r in _classify(bursts, model)]})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-frequency-hz", type=float, help="Declared transmit center for synthetic fixture")
    print(json.dumps(experiment(parser.parse_args().rescue_frequency_hz), indent=2, allow_nan=False))
