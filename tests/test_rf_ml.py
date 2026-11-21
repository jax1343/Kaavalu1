"""Offline checks, discoverable by both pytest and the standard unittest runner."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rf_processing import (classify_signals, detect_signals, extract_features,
                               generate_simulated_disasternet_capture, load_iq_recording)
from src.rf_processing.classifier import MODEL_ID
from src.rf_processing.dataset_generator import FS, TUNING, save_simulated_iq, simulate_recording
from src.rf_processing.feature_extractor import FEATURE_NAMES
from src.rf_processing.iq_loader import analyze_recording


class RFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = simulate_recording("QPSK", 40100, offset_hz=18000)
        cls.bursts, cls.display = analyze_recording(cls.recording)
        cls.row, = extract_features(cls.bursts, FS)

    def test_input_layout_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"x.iq"
            known = np.array([1+2j, 3+4j, 8-1j], dtype="<c8")
            known.tofile(p)
            for fmt in ("complex64", "interleaved_float32"):
                r = load_iq_recording(str(p), FS, TUNING, fmt)
                np.testing.assert_allclose(r["iq_data"], known-known.mean())
                self.assertEqual(r["source_tag"], "RECORDED_IQ")
                self.assertEqual(len(r["sha256"]), 64)
            for bad in (b"", b"1234", b"123456789"):
                p.write_bytes(bad)
                with self.assertRaisesRegex(ValueError, "8-byte"):
                    load_iq_recording(str(p), FS, TUNING)
            np.array([np.nan+0j], dtype="<c8").tofile(p)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_iq_recording(str(p), FS, TUNING)
            for rate, fc, fmt, source in [(0,TUNING,"complex64","SIMULATED"), (FS,np.inf,"complex64","SIMULATED"),
                                          (FS,TUNING,"guess","SIMULATED"), (FS,TUNING,"complex64","LIVE_SDR")]:
                with self.assertRaises(ValueError):
                    load_iq_recording(str(p), rate, fc, fmt, source)

    def test_replay_preserves_simulated_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"replay.npz"
            save_simulated_iq(p, self.recording)
            with self.assertRaisesRegex(ValueError, "npz"):
                load_iq_recording(str(p), FS, TUNING)
            saved = load_iq_recording(str(p), FS, TUNING, fmt="npz")
            rows, display = analyze_recording(saved)
            self.assertEqual((rows[0]["source_tag"], display["source_tag"]), ("SIMULATED","SIMULATED"))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_iq_recording(str(p), FS/2, TUNING, fmt="npz")

    def test_quiet_short_and_unlabelled(self):
        for x in (np.zeros(8192, "complex64"), self.recording["iq_data"][:2047]):
            signals, display = detect_signals(x, FS, TUNING)
            self.assertEqual(signals, [])
            self.assertIsNone(display["source_tag"])
            self.assertTrue(display["frequency_hz"])
        signals, _ = detect_signals(self.recording["iq_data"], FS, TUNING)
        self.assertIsNone(signals[0]["source_tag"])
        with self.assertRaises(ValueError):
            detect_signals(np.arange(2048), FS, TUNING)

    def test_frequency_units_and_simultaneous_emitters(self):
        rng = np.random.default_rng(412)
        t = np.arange(8192)/FS
        x = .06*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        for f in (-24000, 18000):
            x += np.exp(2j*np.pi*f*t)
        signals, _ = detect_signals(x.astype("complex64"), FS, TUNING)
        self.assertEqual(len(signals), 2)
        for row, f in zip(signals, (-24000, 18000)):
            self.assertAlmostEqual(row["center_freq_hz"], TUNING+f, delta=FS/512)
            self.assertGreater(row["occupied_bw_hz"], 0)
            self.assertLess(row["occupied_bw_hz"], 5*FS/512)
            self.assertEqual(row["iq_baseband"].dtype, np.complex64)
            self.assertAlmostEqual(row["duration_s"], len(row["iq_baseband"])/FS)

    def test_separate_bursts_on_same_frequency(self):
        rng = np.random.default_rng(522)
        t = np.arange(16384)/FS
        on = ((t*FS > 1100) & (t*FS < 5300)) | ((t*FS > 9200) & (t*FS < 14100))
        x = on*np.exp(2j*np.pi*18000*t) + .035*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        rows, _ = detect_signals(x, FS, TUNING)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0]["stop_sample"], rows[1]["start_sample"])

    def test_feature_order_and_leakage(self):
        self.assertEqual(tuple(self.row["features"]), ("spectral_flatness","crest_factor_db","amp_std_norm",
                         "amp_kurtosis","phase_std_rad","inst_freq_var_norm","cumulant_c40_mag","cumulant_c42_mag","duty_cycle"))
        bait = dict(self.bursts[0], expected_label="GFSK", scenario_name="test", corridor="rescue",
                    center_freq_hz=1., occupied_bw_hz=2., peak_power_dbm=99.)
        other, = extract_features([bait], FS)
        self.assertEqual(other["features"], self.row["features"])
        a, b = classify_signals([other, self.row])
        for k in ("predicted_modulation","classification_confidence","lof_normality_score"):
            self.assertEqual(a[k], b[k])
        polluted = dict(self.row["features"], expected_label="GFSK")
        self.assertTrue(classify_signals([dict(self.row, features=polluted)])[0]["is_unknown"])

    def test_missing_evidence_and_missing_model(self):
        for changes in ({"snr_db": None},{"snr_db":np.nan},{"sample_count":None},{"sample_count":2047},
                        {"features":{}},{"features":dict(self.row["features"], amp_std_norm=np.nan)}):
            result, = classify_signals([dict(self.row, **changes)])
            self.assertTrue(result["is_unknown"])
            self.assertTrue(result["clf_reason"])
        result, = classify_signals([self.row], {})
        self.assertIn("model", result["clf_reason"])
        self.assertIsNone(result["classification_confidence"])
        rows = extract_features([{"sample_count":2048},{"iq_baseband":np.ones(2048,"complex64"),"sample_count":3000}],FS)
        self.assertTrue(all(r.get("feature_error") for r in rows))

    def test_8db_boundary_and_gate_attribution(self):
        with patch("src.rf_processing.classifier._evaluate", return_value=("QPSK", .68, -.08)) as evaluate:
            low, edge = classify_signals([dict(self.row,snr_db=7.999), dict(self.row,snr_db=8.)], {})
            self.assertEqual(evaluate.call_count, 1)
            self.assertIn("Low SNR", low["clf_reason"])
            self.assertIsNone(low["lof_normality_score"])
            self.assertEqual(edge["predicted_modulation"],"QPSK")
        for score, novelty, expected in ((.67, .5, "Ambiguous"),(.99,-.081,"Novel waveform")):
            with patch("src.rf_processing.classifier._evaluate", return_value=("BPSK", score, novelty)):
                row, = classify_signals([dict(self.row,snr_db=8.)], {})
                self.assertTrue(row["is_unknown"])
                self.assertIn(expected, row["clf_reason"])

    def test_handoff_measurements_and_json(self):
        rows, meta = generate_simulated_disasternet_capture("rescue_corridor")
        features = extract_features(rows, meta["sample_rate_hz"])
        decisions = classify_signals(features)
        self.assertEqual(len(decisions), 1)
        row = decisions[0]
        self.assertEqual(row["source_tag"], "SIMULATED")
        for k in ("signal_id","center_freq_hz","occupied_bw_hz","duration_s","snr_db"):
            self.assertEqual(row[k],rows[0][k])
        self.assertEqual(set(row["features"]),set(FEATURE_NAMES))
        self.assertIsNone(row["peak_power_dbm"])
        self.assertEqual(row["model_id"],MODEL_ID)
        self.assertNotIn("authorization",row)
        self.assertEqual(len(meta["waterfall_db"]),len(meta["time_s"]))
        self.assertEqual(len(meta["waterfall_db"][0]),len(meta["frequency_hz"]))
        self.assertEqual(len(row["constellation_i"]),len(row["constellation_q"]))
        json.dumps([row,meta],allow_nan=False)

    def test_reproducible_and_independent_capture(self):
        twin = simulate_recording("QPSK",40100,offset_hz=18000)
        np.testing.assert_array_equal(twin["iq_data"],self.recording["iq_data"])
        fresh = simulate_recording("QPSK",40101,offset_hz=18000)
        self.assertFalse(np.array_equal(twin["iq_data"],fresh["iq_data"]))
        self.assertNotIn("expected_label",twin)

    def test_signature_weak_slice_and_missed_detection(self):
        rows, _ = generate_simulated_disasternet_capture("familiar_weak")
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0]["snr_db"], 6.)
        self.assertLess(rows[0]["snr_db"], 8.)
        decision, = classify_signals(extract_features(rows, FS))
        self.assertIn("Low SNR", decision["clf_reason"])
        quiet, _ = generate_simulated_disasternet_capture("familiar_clear", added_noise_db=30.)
        self.assertEqual(classify_signals(extract_features(quiet, FS)), [])

    def test_held_out_waveform_preserves_actual_scores(self):
        rows, _ = generate_simulated_disasternet_capture("unfamiliar_clear")
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["snr_db"], 8.)
        decision, = classify_signals(extract_features(rows, FS))
        self.assertTrue(np.isfinite(decision["lof_normality_score"]))
        self.assertTrue(0. <= decision["classification_confidence"] <= 1.)
        self.assertTrue(decision["clf_reason"])
