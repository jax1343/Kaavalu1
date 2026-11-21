# KAAVALU RF validation

On 108 independently seeded synthetic captures, KAAVALU returned the expected open-set label on **103/108**. Narrowband FM accounted for all five errors. Three went to review and two were accepted as GFSK. These results measure waveform recognition on one simulator. Authorization uses configured demonstration rules and has separate tests.

## Reproduce

Use 64-bit Python 3.11 from the repository root. The measured run used Python 3.11.9 and NumPy 1.26.4. The second requirements file pins PyArrow 17 for the existing Streamlit/NumPy combination.

```powershell
python -m pip install -r requirements.txt -r src/dashboard/requirements-local.txt
python -m pip check
$env:KAAVALU_REPLAY_DIR = (Get-Location).Path
python -m pytest -q tests
python check_familiar_clear.py .
python -m src.rf_processing.experiment
python -m src.rf_processing.validation --output validation-run.json
```

The last command builds one model, runs seven scenarios through the backend, scores 108 labelled captures, writes all outcomes and timings, and exits nonzero for a scenario mismatch. It uses a fresh temporary SQLite database. On the Laptop 1 integration checkout, add `--rules config/disasternet_rules.json` to select the configuration explicitly. The saved run used the JSON from integration commit `0a1f43b`. The equivalent five defaults are in `src/backend/rules.py` on main. The evidence contains the complete resolved rules, their digest, the per-capture predictions and the per-case reasons. [Measured data](rf-validation-results.json)

## Pipeline and handoff

**Regression results.** All 28 tests passed on the feature branch based on main commit `f9c7f51`, with the Streamlit replay test enabled. Applying the feature's backend and test files to an isolated copy of Laptop 1's integration commit `0a1f43b` gave 30 passing tests, including its CLI-to-dashboard and rule-only fire demos. The original replay checker also passed. The false-positive rates below concern waveform classification. Operational alert false-positive rates would require independently adjudicated incidents.

| Stage | What the implementation does and what was checked |
|---|---|
| Ingestion | The loader reads little-endian complex64 or interleaved float32 I/Q pairs, or a metadata-bearing NPZ with pickle disabled. It checks receiver metadata and finite complex samples, hashes the source file and removes the complex mean. File replay joins acquisition provenance to each burst. |
| Frequency and bandwidth | The detector slides a 512-sample Hann window in 256-sample steps. The excess-power centroid gives a signed offset from receiver tuning. BW99 spans whole bins between the 0.5% and 99.5% cumulative excess-power points. At 192 ksample/s each bin spans 375 Hz. The two-tone test checks a known 18,562.5 Hz centroid and 2,250 Hz occupied span. |
| SNR and segmentation | The median time-averaged spectral density across receiver bins estimates the background. SNR compares excess power to estimated noise over the active band and frames. Halving voltage at fixed noise reduced measured SNR by about 6.02 dB. Detection requires at least 2,048 samples. The default detection floor is 6 dB. |
| Classification | Each isolated burst supplies nine features. They measure spectral flatness, crest factor, normalized amplitude spread, amplitude kurtosis, circular phase spread, normalized instantaneous-frequency variance, two complex cumulants and duty cycle. Carrier frequency, dBm, occupied bandwidth and scenario names stay outside the vector. The CPU model standardizes features, uses seven distance-weighted neighbours for the candidate and an 18-neighbour Local Outlier Factor for novelty. |
| Decision and units | Below 8 dB the classifier returns UNKNOWN with both scores null. Sufficient-quality inputs must pass novelty (score ≥ −0.08) and weighted vote share (≥ 0.68). The vote share is uncalibrated. `src.contracts.to_shared_record` converts native Hz to shared kHz once. The familiar file produced 9,472 samples, 22.66 dB, and 16,125 Hz → 16.125 kHz through SQLite. |
| Authorization | `src.backend.auth_engine.check_authorization` matches the measured centre frequency to a corridor. At a shared boundary the corridor starting at that frequency wins. Weak, uncertain or unrecognized rows go to UNKNOWN. A recognized row outside all corridors yields RECOGNIZED_NO_AUTH_RECORD. A matched rule requiring unavailable calibrated power yields UNKNOWN before checking other limits. With sufficient evidence, disallowed modulation, bandwidth over 115% of the limit, or power over the limit plus 2 dB yields POTENTIALLY_UNAUTHORIZED. Otherwise the verdict is KNOWN_AUTHORIZED. |
| Storage and API | `save_detected_signals` stores checked rows and alerts transactionally under session and signal IDs. A replay can update the same identity. Missing power is SQL NULL and JSON null. The rule-only demonstration bypasses persistence. The integration entry point is the Python callback `run_kaavalu_pipeline` in Laptop 1's integration branch. The dashboard consumes a classification callback. The inspected repository has no HTTP API. |

**Contract note for Laptop 1.** Existing function signatures, verdict names and SQL tables are preserved. An explicitly `UNCALIBRATED` row carrying a numeric power now has its contradictory number cleared on the backend's working copy. This keeps the policy decision and stored measurement consistent. Legacy numeric measurements without that flag follow the existing contract. Malformed required fields such as absent bandwidth raise a validation error before any verdict is produced.

## Scenario evidence

The generator produces one synthetic emitter per input at 192 ksample/s. Each scenario saves its inputs, expected outcome, actual verdict, classifier reason and source identity. Detector misses produce an empty list. The six IQ cases use measured IQ results and include SQLite readback.

| Scenario | Input or controlled change | Observed result |
|---|---|---|
| Familiar QPSK | Bundled `familiar_clear.npz` at 150 MHz tuning | QPSK, 16.125 kHz. RECOGNIZED_NO_AUTH_RECORD. |
| Authorized QPSK | Seed 81001 at the rescue midpoint. A **test-only** copy of EM-RES-04 has no power ceiling. | QPSK, 16.125 kHz. KNOWN_AUTHORIZED under that test rule. Power remained null and the stored rules retained their original limit. |
| Fire with unavailable power | WB-FM seed 61080 at 148.225 MHz, original fire rule | 19.19 dB, 33 kHz, WB-FM. UNKNOWN because calibrated power is required. |
| Hypothetical fire rule input | Same fire row, with an explicitly hypothetical −20 dBm in memory | POTENTIALLY_UNAUTHORIZED. WB-FM is outside the rule's allowed families and 33 kHz exceeds its 28.75 kHz tolerated limit. The measured source row retained null power in SQLite. |
| Unfamiliar modulation | Held-out 16-QAM seed 91001 | 21.44 dB. The QPSK candidate passed the vote threshold. Novelty caused UNKNOWN. |
| Weak QPSK | Seed 81001, 16 dB extra noise | Detected at 7.60 dB. UNKNOWN at the quality gate. |
| Missed QPSK | Same seed, 18 dB extra noise | Zero detections. No classifier or policy verdict. |

## Classification and latency

Fitting used 48 synthetic sessions for each of six known classes, seeds 10000–10287. Evaluation used seeds 73000–73095 (16 captures per known family) and 96000–96011 (12 unfamiliar 16-QAM captures). No evaluation capture was used to fit or tune the model. All examples share the same simulator recipes as the earlier experiment. One parent capture contributes one result. A missed or multiply detected parent remains in the denominator. Here every evaluation capture produced one detection.

| True family | Correct / total | Precision | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| GFSK | 16/16 | 88.89% | 100% | 2/92 = 2.17% |
| FM-NB | 11/16 | 100% | 68.75% | 0/92 |
| BPSK | 16/16 | 100% | 100% | 0/92 |
| QPSK | 16/16 | 100% | 100% | 0/92 |
| AM-DSB | 16/16 | 100% | 100% | 0/92 |
| WB-FM | 16/16 | 100% | 100% | 0/92 |
| UNKNOWN (true 16-QAM) | 12/12 | 80% | 100% | 3/96 = 3.125% |

Precision is correct calls divided by all calls for a label. Recall is correct calls divided by true examples of that label. Each false-positive rate uses all other true labels as negatives. Known-family accuracy was **91/96 (94.79%)**, including abstentions as errors. FM-NB seeds 73025 and 73031 were called GFSK. Three other FM-NB captures had ambiguous votes. All 12 unfamiliar captures were rejected on novelty, giving **0/12 unfamiliar false accepts** here. On the earlier published independent panel the model accepted 16-QAM seed 92003 as QPSK and correctly identified only 87/96 known captures. Both cohorts matter. [Earlier evaluation](../handoff/observed_results.json)

On the Windows test host, reference fitting took **10.77 s**. With the fitted model, 108 runs from IQ already in memory through detection, feature extraction and classification took **30.86 ms median, 54.37 ms p95**. Each capture had 12,288 samples (64 ms). Timings exclude IQ generation, rule evaluation, storage and UI. Individual scenario timings include generation or file read, classification, policy, SQLite write and readback, with the model already fitted. These measurements cover offline batches on this host. Live throughput, receiver dropouts and end-user dashboard latency still require direct measurement.

## Recorded I/Q feasibility and limits

The offline reader accepts receiver files when the acquisition operator supplies the actual format, sample rate and tuning. Example for a documented receiver capture:

```powershell
python -m src.rf_processing.replay_raw capture.iq --format interleaved_float32 --sample-rate-hz 192000 --center-freq-hz 150000000 --source RECORDED_IQ
```

Use `--source SIMULATED` for generated bytes. The command supports up to one million I/Q pairs, checks complete 8-byte pairs and finite samples, and prints capture provenance plus native classified rows. File replay rejects the LIVE_SDR tag. Tests serialize the supplied simulated capture into both raw formats and replay it through classification and the contract. A file hash identifies bytes; capture provenance still depends on the acquisition record. No hardware-labelled receiver accuracy was measured.

Future receiver validation needs a receive-capable SDR covering the band and bandwidth, a suitable antenna, documented gain and tuning, and a reference source for absolute dBm calibration. Retain acquisition metadata, time, calibration evidence and independently established waveform labels. A live acquisition adapter must demonstrate continuous input, sample continuity and sustained processing under load. The ESP32 alert output cannot provide the arbitrary-band IQ used here.

The noise estimator assumes much of the sampled spectrum is background. Crowded channels, overlapping emitters, very short bursts, band-edge signals, clipping and oscillator drift need separate measurement. Centre-based corridor matching does not assess full spectral overlap. A single unfamiliar family and a single synthetic emitter per capture leave field false-alarm rates and transmitter identity unmeasured.
