"""From the project root run python -m pytest -q tests/test_backend_auth.py."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.backend import check_authorization, generate_alerts, get_auth_rules, init_db, save_detected_signals
from src.backend.db_manager import (insert_coordinator_review, load_session, review_history,
                                    save_rules_and_results)
from src.contracts import power_display_text


def signal(**fields):
    row = dict(signal_id="burst", capture_id="capture", source_tag="SIMULATED",
               center_freq_hz=148500000, occupied_bw_hz=16125, peak_power_dbm=-15.0,
               snr_db=22.66, classification_confidence=1.0, predicted_modulation="QPSK",
               is_unknown=False, features={"duty_cycle": .8}, constellation_i=[.5], constellation_q=[.1])
    row.update(fields)
    return row


class BackendTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = str(Path(temp.name) / "test.db")
        init_db(self.db)
        self.rules = get_auth_rules(self.db)

    def check(self, **changes):
        return check_authorization([signal(**changes)], self.rules)[0]

    def test_demo_boundaries_and_tolerances(self):
        self.assertEqual(len(self.rules), 5)
        for hz, key in [(148150000, "EM-FIR-02"), (148850000, "EM-AIR-05")]:
            self.assertEqual(self.check(center_freq_hz=hz)["matched_rule_id"], key)
        self.assertEqual(self.check(occupied_bw_hz=57500, peak_power_dbm=-8)["verdict_category"], "KNOWN_AUTHORIZED")
        row = self.check(occupied_bw_hz=57501, peak_power_dbm=0.0, predicted_modulation="WB-FM")
        self.assertEqual(row["verdict_category"], "POTENTIALLY_UNAUTHORIZED")
        self.assertEqual(row["alert_severity"], "CRITICAL")
        for word in ["bandwidth", "power", "WB-FM"]:
            self.assertIn(word, row["explainable_reason"])

    def test_fairness_precedes_rule_mismatch(self):
        for changes in [dict(is_unknown=True), dict(is_unknown=None), dict(predicted_modulation="UNKNOWN"),
                        dict(predicted_modulation="unfamiliar"), dict(snr_db=7.99), dict(snr_db=None),
                        dict(classification_confidence=.679), dict(classification_confidence=None)]:
            with self.subTest(changes=changes):
                self.assertEqual(self.check(peak_power_dbm=0, **changes)["verdict_category"], "UNKNOWN")
        self.assertEqual(self.check(snr_db=8, classification_confidence=.68)["verdict_category"], "KNOWN_AUTHORIZED")

    def test_power_depends_on_matched_rule(self):
        self.assertEqual(self.check(peak_power_dbm=None)["verdict_category"], "UNKNOWN")
        self.assertEqual(self.check(peak_power_dbm=None, center_freq_hz=150017956)["verdict_category"],
                         "RECOGNIZED_NO_AUTH_RECORD")
        self.rules[3]["max_power_dbm"] = None
        self.assertEqual(self.check(peak_power_dbm=None)["verdict_category"], "KNOWN_AUTHORIZED")

    def test_recurrence_scope_and_alerts(self):
        rows = [signal(is_unknown=True), signal(signal_id="two", center_freq_hz=148520000, is_unknown=True),
                signal(capture_id="other", is_unknown=True)]
        checked = check_authorization(rows, self.rules)
        self.assertEqual([r["recurrence_count"] for r in checked], [2, 2, 1])
        alerts = generate_alerts(checked + [self.check()], {})
        self.assertEqual([a["severity"] for a in alerts], ["HIGH", "HIGH", "MEDIUM"])
        self.assertEqual(len({a["alert_id"] for a in alerts}), 3)

    def test_units_provenance_and_validation(self):
        native = signal()
        once = check_authorization([native], self.rules)[0]
        twice = check_authorization([once], self.rules)[0]
        self.assertEqual(once, twice)
        self.assertEqual(twice["occupied_bw_khz"], 16.125)
        self.assertNotIn("occupied_bw_khz", native)
        for bad in [dict(occupied_bw_khz=.016125), dict(source_tag=""), dict(occupied_bw_hz=float("nan")),
                    dict(center_freq_mhz=200)]:
            with self.assertRaises(ValueError):
                self.check(**bad)

    def test_persistence_retry_and_review(self):
        row = self.check(peak_power_dbm=None)
        save_detected_signals([row, row], self.db)
        read, = load_session("capture", self.db)
        self.assertEqual(read["source_tag"], "SIMULATED")
        self.assertEqual(power_display_text(read["peak_power_dbm"]), "Power unavailable.")
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute("SELECT bandwidth_hz, peak_power_dbm FROM signal_detections").fetchall(),
                             [(16125, None)])
            self.assertEqual(db.execute("SELECT count(*) FROM generated_alerts").fetchone()[0], 1)
        insert_coordinator_review("capture", "burst", "Operator", "Coordinator", "Needs follow-up", "", self.db)
        clear = self.check()
        save_detected_signals([clear], self.db)
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM generated_alerts").fetchone()[0], 0)
        self.assertEqual(len(review_history("capture", self.db)), 1)

    def test_atomic_batch_and_rule_changes(self):
        valid = self.check()
        with self.assertRaises(ValueError):
            save_detected_signals([valid, dict(valid, peak_power_dbm="bad")], self.db)
        self.assertEqual(load_session("capture", self.db), [])
        with self.assertRaises(ValueError):
            save_rules_and_results([], [dict(valid, occupied_bw_khz=-1)], self.db)
        self.assertEqual(get_auth_rules(self.db), self.rules)
        with self.assertRaises(ValueError):
            save_rules_and_results(self.rules + [self.rules[0]], [], self.db)
        save_rules_and_results([], [], self.db)
        init_db(self.db)
        self.assertEqual(get_auth_rules(self.db), [])

    def test_streamlit_review_and_rule_editor(self):
        from streamlit.testing.v1 import AppTest
        save_detected_signals([self.check(peak_power_dbm=None)], self.db)
        app = Path(__file__).resolve().parents[1] / "src/dashboard/app.py"
        with patch.dict(os.environ, KAAVALU_DB_PATH=self.db):
            ui = AppTest.from_file(str(app)).run(timeout=30)
            self.assertFalse(ui.exception)
            self.assertIn("Power unavailable.", [m.value for m in ui.metric])
            self.assertIn("SIMULATED", [s.value for s in ui.subheader])
            inputs = {t.label: t for t in ui.text_input}
            inputs["Coordinator name"].input("Operator")
            inputs["Coordinator role"].input("SAR coordinator")
            next(b for b in ui.button if b.label == "Save review").click().run()
            self.assertEqual(len(review_history("capture", self.db)), 1)
            # AppTest 1.37 exposes the editor as a dataframe. Submit its widget
            # payload alongside the save click, as the Streamlit client does.
            editor = next(f for f in ui.dataframe if f.proto.id.endswith("-rule_editor"))
            next(b for b in ui.button if b.label == "Save & Re-Evaluate").click()
            states = ui._tree.get_widget_states()
            states.widgets.add(id=editor.proto.id, string_value=json.dumps({
                "edited_rows": {"3": {"max_power_dbm": None}}, "added_rows": [], "deleted_rows": []}))
            ui._run(states)
            self.assertFalse(ui.exception)
            self.assertEqual(load_session("capture", self.db)[0]["verdict_category"], "KNOWN_AUTHORIZED")

    @unittest.skipUnless(os.environ.get("KAAVALU_REPLAY_DIR"), "Set KAAVALU_REPLAY_DIR to Laptop 2's extracted ZIP")
    def test_actual_laptop2_to_streamlit(self):
        from streamlit.testing.v1 import AppTest
        import numpy as np
        self.assertEqual(sys.version_info[:2], (3, 11))
        self.assertEqual(np.__version__, "1.26.4")
        folder = Path(os.environ["KAAVALU_REPLAY_DIR"])
        output = subprocess.check_output([sys.executable, "-B", "replay_familiar_clear.py"], cwd=folder)
        native = json.loads(output)
        row, = check_authorization([native], self.rules)
        self.assertEqual(row["verdict_category"], "RECOGNIZED_NO_AUTH_RECORD")
        self.assertEqual(row["predicted_modulation"], "QPSK")
        self.assertEqual(row["occupied_bw_khz"], 16.125)
        save_detected_signals([row], self.db)
        self.assertIsNone(load_session(row["session_id"], self.db)[0]["peak_power_dbm"])
        app = Path(__file__).resolve().parents[1] / "src/dashboard/app.py"
        with patch.dict(os.environ, KAAVALU_DB_PATH=self.db):
            ui = AppTest.from_file(str(app)).run(timeout=30)
            self.assertFalse(ui.exception)
            self.assertIn("Power unavailable.", [m.value for m in ui.metric])
            self.assertIn("16.125 kHz", [m.value for m in ui.metric])
            self.assertEqual(ui.selectbox[0].value, row["signal_id"])


if __name__ == "__main__":
    unittest.main()
