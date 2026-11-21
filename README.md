# KAAVALU local RF integration handoff

Use Python 3.11 and NumPy 1.26.4. The input for this handoff is simulated IQ. The standalone classifier builds its reference model locally on first use.

## Run Laptop 1's entry point

From the checkout root in a Python 3.11 environment:

    python -m pip install -r requirements.txt
    python main.py --cli familiar_clear
    python -m streamlit run main.py --server.address 127.0.0.1 --browser.gatherUsageStats false

Select **Run analysis** in the dashboard. Both entry points read the bundled
`fixtures/familiar_clear.npz`. CLI output identifies the SQLite database,
session, signal, model, matching rules and verdict. The full record is in
`signal_detections.record_json` in `kaavalu.db`. Set `KAAVALU_DB_PATH` to
choose a different database. A fresh database loads the five demonstration
corridors from `config/disasternet_rules.json`. An existing database retains
the coordinator's rule edits.

For the fire-corridor demonstration run:

    python main.py --cli fire_wide --rule-only

The entry point generates seeded simulated IQ and saves its measured result.
An additional in-memory rule-only example uses the same measured width and
modulation with a hypothetical power of −20 dBm. The output labels that
example separately. The measured record retains its unavailable power.

## Verify the standalone replay

From the repository root, use the team environment and run:

    python replay_familiar_clear.py
    python check_familiar_clear.py .

The saved outgoing dictionary is in `handoff/familiar_clear_outgoing.json`. The shared measurement adapter in `src/contracts.py` converts the native 16125 Hz field to 16.125 kHz and preserves unavailable calibrated power.

## Backend and dashboard

The RF-row consumer is `src/backend/auth_engine.py`. The checked-in rule JSON
uses the same five demonstration corridors as the backend defaults.

For the dashboard use `src/dashboard/START-HERE.md`. Its local dependencies include PyArrow 17.0.0 for the team's NumPy pin. To exercise both suites in PowerShell with the pinned dashboard dependencies installed:

    $env:KAAVALU_REPLAY_DIR = (Get-Location).Path
    python -m pytest -q tests

The master handover and Part 2 RF/ML guide are under `docs/`. The original replay ZIP and combined integration handoff ZIP are included for direct download.
