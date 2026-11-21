# KAAVALU Laptop 2 replay handoff

Extract the ZIP before running anything. You can open the JSON files in a text editor without installing Python.

| File | Contents |
|---|---|
| fixtures/familiar_clear.npz | Original simulated input: 12,288 complex64 IQ samples, 192,000 samples/s, receiver tuning 150,000,000 Hz and embedded SIMULATED provenance. |
| handoff/familiar_clear_outgoing.json | Full, unedited classified dictionary for one burst produced by loading and replaying the included file. Includes identifiers, nine features, measurements, constellation, scores, reason and null calibrated power. |
| handoff/familiar_clear_display.json | Separate full detector display metadata for the same replay, including the spectrum and waterfall. |
| handoff/familiar_clear_direct_outgoing.json | Full classified dictionary from the original in-memory familiar_clear scenario. File replay hashes the archive and removes the recorded mean, so identifiers and some numbers differ between these two routes. |
| src/rf_processing/ and tests/test_rf_ml.py | The standalone RF project and its tests. |
| handoff/LAPTOP_1.md and handoff/observed_results.json | Interface notes and results of the broader synthetic experiment. |

## Replay on Windows

From PowerShell in the extracted directory, with the team Python environment active:

```powershell
python replay_familiar_clear.py
```

The script loads the saved IQ, detects the burst, computes nine waveform features, trains the default synthetic reference model locally on first use, and prints the classifier's first and only row. The printed JSON is what Laptop 1 can compare against its shared contract. The replay needs Python 3.11 or newer and NumPy. The project targets NumPy 1.26.4 on Python 3.11. Replay was verified with Python 3.12.14 and NumPy 2.3.5.

The packet records the current standalone output. Laptop 1 still needs to verify compatibility with its actual contracts and policy caller. Calibrated power is unavailable. The classification confidence is a weighted neighbor vote share.
