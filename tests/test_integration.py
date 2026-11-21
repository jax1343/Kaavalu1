"""Laptop 1 entry points using real simulated IQ, team modules and SQLite."""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ID = "a0beec8d5fff306f677d/b1"


def cli(database, *args):
    output = subprocess.check_output(
        [sys.executable, "main.py", "--cli", *args], cwd=ROOT,
        env=dict(os.environ, KAAVALU_DB_PATH=str(database)),
    )
    return json.loads(output)


def test_familiar_from_cli_to_dashboard(tmp_path, monkeypatch):
    database = tmp_path / "capture.db"
    run = cli(database, "familiar_clear")
    signal, = run["records"]
    assert signal["signal_id"] == EXPECTED_ID
    assert signal["sha256"] == hashlib.sha256((ROOT / "fixtures/familiar_clear.npz").read_bytes()).hexdigest()
    assert signal["model_id"] == "kaavalu-synthetic-knn-lof-v1"
    assert signal["predicted_modulation"] == "QPSK" and round(signal["snr_db"], 2) == 22.66
    assert signal["occupied_bw_khz"] == 16.125 and signal["peak_power_dbm"] is None
    assert run["matched_rules"] == [] and signal["matched_rule_id"] is None
    assert signal["verdict_category"] == "RECOGNIZED_NO_AUTH_RECORD"

    monkeypatch.setenv("KAAVALU_DB_PATH", str(database))
    dashboard = AppTest.from_file(str(ROOT / "main.py")).run(timeout=45)
    assert not dashboard.exception
    next(b for b in dashboard.button if b.label == "Run analysis").click().run(timeout=45)
    assert not dashboard.exception and not dashboard.error
    assert len(dashboard.get("plotly_chart")) == 2
    assert next(box.value for box in dashboard.selectbox if box.label == "Inspect signal") == EXPECTED_ID
    assert "SIMULATED" in [header.value for header in dashboard.subheader]
    assert "Power unavailable." in [metric.value for metric in dashboard.metric]
    assert "16.125 kHz" in [metric.value for metric in dashboard.metric]
    with closing(sqlite3.connect(database)) as db:
        payload, power = db.execute("SELECT record_json,peak_power_dbm FROM signal_detections").fetchone()
        stored = json.loads(payload)
        assert {key: stored[key] for key in signal} == signal and power is None
        assert db.execute("SELECT signal_id,alert_title FROM generated_alerts").fetchall() == [
            (EXPECTED_ID, signal["verdict_category"])]


def test_fire_iq_and_prepared_rule_only_are_distinct(tmp_path):
    database = tmp_path / "fire.db"
    report = cli(database, "fire_wide", "--rule-only")
    measured, = report["records"]
    rule, = report["matched_rules"]
    assert rule["rule_id"] == measured["matched_rule_id"] == "EM-FIR-02"
    assert rule["max_bw_khz"] == 25 and rule["max_power_dbm"] == -15
    assert measured["predicted_modulation"] == "WB-FM" and measured["occupied_bw_khz"] == 33
    assert measured["verdict_category"] == "UNKNOWN" and measured["peak_power_dbm"] is None
    example = report["rule_only_example"]
    assert example["label"] == "PREPARED RULE-ONLY TEST"
    assert example["derived_from_signal_id"] == measured["signal_id"]
    assert example["record"]["peak_power_dbm"] == -20
    assert example["record"]["verdict_category"] == "POTENTIALLY_UNAUTHORIZED"
    assert "bandwidth 33 kHz exceeds 28.75 kHz" in example["record"]["explainable_reason"]
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT signal_id,peak_power_dbm,verdict_category FROM signal_detections").fetchall() == [
            (measured["signal_id"], None, "UNKNOWN")]
