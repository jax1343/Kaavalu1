"""Dashboard-owned contract and operator workflow checks on a fresh SQLite DB."""
import csv
import io
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.backend.db_manager import get_auth_rules, load_session, review_history
from src.dashboard.incidents import evidence_csv, incident
from src.dashboard.monitor import spectrum
from src.dashboard.scenarios import AUTHORIZED, FAMILIAR, FIRE, HYPOTHETICAL, UNKNOWN


ROOT = Path(__file__).resolve().parents[1]


def test_all_scenarios_and_review_export(tmp_path, monkeypatch):
    db = str(tmp_path / "incidents.db")
    monkeypatch.setenv("KAAVALU_DB_PATH", db)
    app = AppTest.from_file(str(ROOT / "src/dashboard/app.py")).run(timeout=45)
    for label, verdict in [(FAMILIAR, "RECOGNIZED_NO_AUTH_RECORD"), (AUTHORIZED, "KNOWN_AUTHORIZED"),
                           (UNKNOWN, "UNKNOWN"), (HYPOTHETICAL, "POTENTIALLY_UNAUTHORIZED"), (FIRE, "UNKNOWN")]:
        next(r for r in app.radio if r.label == "Input scenario").set_value(label).run(timeout=45)
        next(b for b in app.button if b.label == "Run analysis").click().run(timeout=45)
        assert not app.exception and not app.error
        assert any(f"<strong>{verdict}</strong>" in m.value for m in app.markdown)
        assert "SIMULATED" in [s.value for s in app.subheader]
        if label in (AUTHORIZED, HYPOTHETICAL):
            assert not any(b.label == "Save review" for b in app.button)
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT count(*) FROM signal_detections").fetchone() == (3,)
        assert conn.execute("SELECT count(*) FROM signal_detections WHERE peak_power_dbm IS NOT NULL").fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM authorization_rules").fetchone() == (5,)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Measured width"] == "33.00 kHz"
    assert metrics["Adjusted limit"] == "28.75 kHz"
    assert metrics["Excess bandwidth"] == "4.25 kHz"
    assert metrics["Calibrated peak power"] == "Power unavailable."
    session = next(s for s in app.selectbox if s.label == "Stored session").value
    for status in ("Verified", "False alarm", "Escalate"):
        next(t for t in app.text_input if t.label == "Coordinator name").input("Test operator")
        next(t for t in app.text_input if t.label == "Coordinator role").input("Fire coordinator")
        next(r for r in app.radio if r.label == "Disposition").set_value(status)
        next(b for b in app.button if b.label == "Save review").click().run(timeout=45)
        assert not app.exception and not app.error
    history = review_history(session, db)
    assert [r["disposition"] for r in history] == ["Escalate", "False alarm", "Verified"]
    records, rules = load_session(session, db), get_auth_rules(db)
    lines = list(csv.DictReader(io.StringIO(evidence_csv(records, rules, history))))
    assert [r["decision"] for r in lines] == ["Escalate", "False alarm", "Verified"]
    assert all(r["verdict_category"] == "UNKNOWN" and r["review_status"] == "Escalate" for r in lines)
    assert all(r["capture_timestamp"] == r["peak_power_dbm"] == "" for r in lines)
    assert all(r["evaluated_at_utc"] and r["reviewed_at_utc"] and r["operator"] == "Test operator" for r in lines)
    assert all(r["matched_rule_id"] == "EM-FIR-02" and r["adjusted_limit_khz"] == "28.75" for r in lines)
    # A different session's review or later rule edit must not rewrite historical evidence.
    changed = [dict(r, max_bw_khz=60) for r in rules]
    assert incident(records[0], changed, [])["adjusted_limit_khz"] == 28.75
    legacy = {k:v for k,v in records[0].items() if not k.startswith("dashboard_")}
    assert incident(legacy, changed, [])["rule_basis"] == "Current configuration only"


def test_strength_uses_measured_time_bins_only():
    meta = dict(spectrum_freq_hz=[149e6,150e6,151e6], spectrum_db=[-5,-3,-7],
                spectrum_unit="dB re 1 IQ-unit²/Hz", frequency_hz=[149e6,150e6,151e6],
                time_s=[.1,.2], waterfall_db=[[-12,-4,-8],[-31,-16,-5]])
    signal = dict(center_freq_mhz=150, occupied_bw_khz=20, verdict_category="UNKNOWN")
    fig = spectrum(meta,[signal],signal,[])
    np.testing.assert_array_equal(fig.data[-1].y, [-4,-16])
    np.testing.assert_array_equal(fig.data[-1].x, meta["time_s"])
    assert len(fig.layout.shapes) == 1
    assert len(spectrum(dict(meta, waterfall_db=[]),[signal],signal,[]).data) == 1
    assert spectrum({},[signal],signal,[]) is None
