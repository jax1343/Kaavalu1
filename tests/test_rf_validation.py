from contextlib import closing
import hashlib
import sqlite3

import numpy as np
import pytest

from src.backend import check_authorization, get_auth_rules, init_db
from src.backend.rules import demo_rules
from src.rf_processing import detect_signals
from src.rf_processing.classifier import train_reference_model
from src.rf_processing.dataset_generator import FS, TUNING
from src.rf_processing.replay_raw import replay_raw
from src.rf_processing.validation import FIXTURE, run_scenarios, score_trials


def test_two_tone_measurements_have_physical_reference():
    # Equal tones three FFT bins apart, on bin centres. Hann main lobes
    # contribute the adjacent bins on both sides to the 99% power interval.
    t = np.arange(12288) / FS
    rng = np.random.default_rng(190)
    noise = .02 * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    tones = np.exp(2j*np.pi*18000*t) + np.exp(2j*np.pi*19125*t)
    snr = []
    for amplitude in (1., .5):
        iq = (amplitude * tones + noise).astype("complex64")
        rows, _ = detect_signals(iq, FS, TUNING)
        row, = rows
        assert row["center_freq_hz"] == pytest.approx(TUNING + 18562.5, abs=2, rel=0)
        assert row["occupied_bw_hz"] == 6 * FS / 512
        snr.append(row["snr_db"])
    assert snr[0] - snr[1] == pytest.approx(20*np.log10(2), abs=.05)


@pytest.mark.parametrize("fmt", ["complex64", "interleaved_float32"])
def test_raw_file_replay_end_to_end(tmp_path, fmt):
    with np.load(FIXTURE, allow_pickle=False) as data:
        iq = data["iq_data"]
    path = tmp_path / "synthetic.raw"
    if fmt == "interleaved_float32":
        np.column_stack([iq.real, iq.imag]).astype("<f4").tofile(path)
    else:
        iq.astype("<c8").tofile(path)
    capture = replay_raw(path, FS, TUNING, fmt, "SIMULATED")
    row, = capture["records"]
    assert capture["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert row["source_tag"] == "SIMULATED" and row["sample_count"] == 9472
    assert row["predicted_modulation"] == "QPSK" and row["peak_power_dbm"] is None
    assert row["occupied_bw_hz"] == 16125
    assert check_authorization([row], demo_rules())[0]["occupied_bw_khz"] == 16.125
    with pytest.raises(ValueError, match="origin"):
        replay_raw(path, FS, TUNING, fmt, "LIVE_SDR")


def test_raw_rejects_partial_and_oversize_files(tmp_path):
    path = tmp_path / "incomplete.raw"
    path.write_bytes(b"1234")
    with pytest.raises(ValueError, match="8-byte"):
        replay_raw(path, FS, TUNING, "complex64", "SIMULATED")
    with path.open("wb") as stream:
        stream.truncate(8_000_008)
    with pytest.raises(ValueError, match="1,000,000"):
        replay_raw(path, FS, TUNING, "complex64", "SIMULATED")


def test_scenarios_and_sqlite(tmp_path):
    db = str(tmp_path / "cases.db")
    init_db(db)
    rules = get_auth_rules(db)
    results = run_scenarios(train_reference_model(), db, rules)
    assert set(results) == {"familiar_clear", "authorized_qpsk", "fire_wide", "fire_rule_only",
                            "unfamiliar_clear", "weak_qpsk", "missed_qpsk"}
    assert all(case["passed"] for case in results.values()), results
    assert get_auth_rules(db) == rules
    familiar, = results["familiar_clear"]["records"]
    assert familiar["signal_id"] == "a0beec8d5fff306f677d/b1"
    assert familiar["occupied_bw_khz"] == 16.125
    weak, = results["weak_qpsk"]["records"]
    assert weak["snr_db"] < 8 and weak["classification_confidence"] is None
    novel, = results["unfamiliar_clear"]["records"]
    assert novel["snr_db"] > 8 and novel["lof_normality_score"] < -.08
    fire, = results["fire_wide"]["records"]
    assert fire["occupied_bw_khz"] == 33 and "Power unavailable" in fire["explainable_reason"]
    hypothetical, = results["fire_rule_only"]["records"]
    assert "28.75 kHz" in hypothetical["explainable_reason"]
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT count(*), count(peak_power_dbm) FROM signal_detections").fetchone() == (5, 0)
        assert conn.execute("SELECT count(*) FROM signal_detections WHERE signal_id = ?",
                            (hypothetical["signal_id"],)).fetchone() == (0,)


def test_missing_or_contradictory_evidence_cannot_authorize():
    rules = demo_rules()
    row = dict(source_tag="SIMULATED", signal_id="b1", capture_id="fixture",
               center_freq_hz=148550000, occupied_bw_hz=16125, peak_power_dbm=-20,
               calibration_status="UNCALIBRATED", predicted_modulation="QPSK",
               is_unknown=False, classification_confidence=1.0, snr_db=22)
    for power in (None, -20., 0.):
        checked, = check_authorization([dict(row, peak_power_dbm=power)], rules)
        assert checked["peak_power_dbm"] is None
        assert checked["verdict_category"] == "UNKNOWN" and "Power unavailable" in checked["explainable_reason"]
    for field in ("occupied_bw_hz", "peak_power_dbm", "center_freq_hz"):
        broken = row.copy()
        del broken[field]
        with pytest.raises(ValueError):
            check_authorization([broken], rules)


def test_metrics_keep_misses_abstentions_and_false_acceptances():
    trials = [dict(family=truth, prediction=label) for truth, label in (
        ("QPSK","QPSK"), ("QPSK","UNKNOWN"), ("QPSK","MISSED"), ("BPSK","MULTIPLE"),
        ("16-QAM","QPSK"), ("16-QAM","UNKNOWN"))]
    scores = score_trials(trials)
    qpsk = scores["per_class"]["QPSK"]
    assert qpsk["precision"] == .5 and qpsk["recall"] == 1/3 and qpsk["false_positive_rate"] == 1/3
    assert scores["known_accuracy"] == .25 and scores["overall_accuracy"] == 2/6
    assert scores["unfamiliar_accepted"] == scores["known_sent_to_review"] == 1
