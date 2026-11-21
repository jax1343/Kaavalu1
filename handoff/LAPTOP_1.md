# KAAVALU Laptop 2 handoff

All five public interfaces are implemented in this standalone candidate. The team checkout, current shared contracts, corridor JSON, and integration caller were unavailable. Compare this package against the existing rf-processing branch before applying changes. The two integration directories are src/rf_processing/ and tests/. The handoff/ directory is for the team leader.

## Run the 90-second demonstration

From the extracted folder with the project environment active, run:

```powershell
python -m src.rf_processing.experiment
```

The experiment prints measured results as JSON, including every held-out prediction. A previously observed run is in handoff/observed_results.json. Simulations use 12,288 complex samples per capture at 192,000 samples/s.

| Demonstration | Measured SNR | Observed result |
|---|---:|---|
| Clear QPSK | 22.66 dB | QPSK, vote share 1.000, LOF +0.370 |
| Same QPSK with 16.0 dB added noise | 7.60 dB | Detected. UNKNOWN because measured SNR fell below 8.0 dB. No model score was calculated. |
| Same QPSK with 18.0 dB added noise | Unavailable | First missed detection on the 0.5 dB noise sweep. Zero rows reached classification. |
| Clear held-out 16-QAM, seed 91001 | 21.44 dB | QPSK candidate, vote share 0.867, LOF −0.428. Novelty check returned UNKNOWN. |
| Held-out 16-QAM, seed 92003 | 18.81 dB | False acceptance as QPSK, vote share 1.000, LOF +0.087. Both gates passed. |
| Illustrative simulated rescue handoff | 24.47 dB | Measured center 150,018,001.75 Hz, BW99 7,875 Hz, FM-NB, vote share 1.000. |

The rescue frequency was chosen for an isolated fixture. Use the actual configured corridor center as the generator setting when the rule file is available, e.g. pass its numeric value after `--rescue-frequency-hz`. That option sets the simulated transmitter. The detector computes the output frequency and bandwidth from IQ. Laptop 3 owns the rule comparison.

## Public imports and caller

```python
from src.rf_processing import (
    load_iq_recording, detect_signals, extract_features, classify_signals,
    generate_simulated_disasternet_capture,
)

signals, meta = generate_simulated_disasternet_capture("familiar_clear")
row, = classify_signals(extract_features(signals, meta["sample_rate_hz"]))
print(row["source_tag"], row["predicted_modulation"], round(row["snr_db"], 2))
# Observed: SIMULATED QPSK 22.66
```

For genuine file replay, call `recording = load_iq_recording(path, sample_rate_hz, center_freq_hz)`, then `from src.rf_processing.iq_loader import analyze_recording` and `signals, meta = analyze_recording(recording)`. Continue with the same feature and classifier calls. A direct call to detect_signals returns source_tag=None. The acquisition caller attaches the source and capture identity. analyze_recording does that join. Use LIVE_SDR only on a recording dictionary constructed by the real live acquisition adapter.

The loader accepts little-endian raw complex64 and interleaved_float32. Headerless bytes require an explicit source when their origin is simulated. For a self-describing simulation replay, use `save_simulated_iq(path, recording)` from dataset_generator and load with `fmt="npz"`. The archive carries source and tuning. Loading removes the complex DC mean. It rejects invalid samples and computes SHA-256 over the original file.

## Provisional data contract

| Stage | Exact fields and units |
|---|---|
| Recording | iq_data (1-D complex64), sample_rate_hz (complex samples/s), center_freq_hz (receiver tuning, Hz), source_tag, sample_count, sha256 |
| Detector row after provenance join | signal_id, capture_id, source_tag, sha256, iq_baseband (isolated complex64, at original sample rate), sample_rate_hz, tuning_freq_hz, offset_hz, center_freq_hz (measured RF center), occupied_bw_hz (BW99), snr_db, start_sample (inclusive), stop_sample (exclusive), sample_count, start_s, duration_s, peak_power_dbm (null), calibration_status |
| Display | spectrum_freq_hz, spectrum_db, noise_floor_db_per_hz, spectrum_unit, frequency_hz, time_s, waterfall_db (time × frequency, at most 160 × 256). Spectrum and waterfall express relative digital power density. Per-slice constellation_i and constellation_q contain up to 256 paired baseband samples in digital units. |
| Extractor | Measurement and identity fields plus features, an ordered dictionary, and feature_error on invalid input. Complex IQ is omitted from the classified JSON. |
| Classifier | predicted_modulation, candidate_modulation, classification_confidence (0–1 weighted neighbor vote share), lof_normality_score (dimensionless), clf_reason, is_unknown, model_id, rule_family, features and retained measurements. Uncomputed scores are null. Provisional rule_family mirrors the accepted waveform label or UNKNOWN. |

The nine feature names in order are spectral_flatness, crest_factor_db, amp_std_norm, amp_kurtosis, phase_std_rad, inst_freq_var_norm, cumulant_c40_mag, cumulant_c42_mag, duty_cycle. Input is built only from this allowlist. Frequency, bandwidth, power and scenario labels never enter the model.

**Policy integration issue.** peak_power_dbm is null and calibration_status is UNCALIBRATED. Laptop 1 must confirm that contracts, storage and policy accept null or explicitly bypass power comparisons when calibration is unavailable. Never substitute zero dBm. Confirm rule_family mapping, display field names and provenance plumbing against the team's source before connecting the dashboard.

## How measurements and decisions work

A 512-sample Hann window slides in 256-sample steps. The median of the time-averaged spectral densities estimates the background noise per Hz. Strong bands seed activity, and lower thresholds trace its time span. SNR compares excess linear power with estimated noise across the candidate's frequency bins and active frames. BW99 spans the bins between 0.5% and 99.5% of cumulative excess power. A power-weighted centroid gives the signed offset from receiver tuning. These estimates depend on the chosen band and the assumption that most of the receiver band contains background.

With model_bundle=None, the first eligible call trains a seeded, CPU-only reference model on 288 independent synthetic sessions and caches it for that process. The model uses standardized features, seven distance-weighted neighbors for classification and an 18-neighbor Local Outlier Factor for familiarity. This is a NumPy implementation. The confidence value is an uncalibrated vote share. A missing or incompatible supplied bundle returns a review reason. SNR below 8.0 dB or fewer than 2,048 samples triggers review before inference. Exactly 8.0 dB continues. LOF below −0.08 triggers novelty review. Otherwise a vote share below 0.68 triggers confidence review. These cutoffs predate the held-out 16-QAM trials.

## Independent evaluation

Training used 48 sessions per class, seeds 10000–10287. Known-class development used separate seeds. Evaluation used 16 new sessions per class, seeds 61000–61095. Each capture contained one emitter. No burst windows from one parent capture crossed the training/test split. All six classes shared the same tuning and offset-generation procedure. Test and training captures share the simulator's waveform recipes.

| True waveform | Correct | Sent for review | Wrong known label | Missed |
|---|---:|---:|---:|---:|
| GFSK | 16 | 0 | 0 | 0 |
| FM-NB | 11 | 3 | 2 as GFSK | 0 |
| BPSK | 14 | 2 | 0 | 0 |
| QPSK | 16 | 0 | 0 | 0 |
| AM-DSB | 16 | 0 | 0 | 0 |
| WB-FM | 14 | 2 | 0 | 0 |
| **Total** | **87** | **7** | **2** | **0** |

All 12 batch-evaluation 16-QAM captures (seeds 92000–92011) were detected above 8 dB. Eleven were rejected by novelty and one was accepted as QPSK. The separate showcase seed 91001 was also rejected by novelty. No 16-QAM data was used for training or threshold selection. No receiver recordings were evaluated.

## Verification

Project command for the pinned Windows Python 3.11 environment:

```powershell
python -m pytest -q tests/test_rf_ml.py
python -m src.rf_processing.experiment
```

In the isolated environment, Python 3.11.9 compiled the source. Its environment lacked the pinned packages, and the package index was unreachable. The 12 test methods ran and passed with Python 3.12.14, NumPy 2.3.5 and the standard library runner:

```powershell
python -m unittest discover -s tests -p test_rf_ml.py -v
```

The detector and model are intended for offline CPU use and use NumPy and the standard library. Verification on all team pins and the real shared contract is pending. Remaining measurement limits include crowded captures, overlapping emitters inside one detected band, short bursts, frequency bands crossing the sampled band edge and long captures that exceed RAM. The waveform simulation omits many receiver impairments. No authorization or legal determination is produced.
