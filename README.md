# KAAVALU local RF integration handoff

The [RF validation report](docs/rf-validation.md) includes the offline scenario runner, measured classification and timing results, recorded I/Q replay commands and the contract handoff for Laptop 1.

Use Python 3.11 and NumPy 1.26.4. The input for this handoff is simulated IQ. The standalone classifier builds its reference model locally on first use.

## Verify the replay

From the repository root, use the team environment and run:

    python replay_familiar_clear.py
    python check_familiar_clear.py .

The saved outgoing dictionary is in `handoff/familiar_clear_outgoing.json`. The shared measurement adapter in `src/contracts.py` converts the native 16125 Hz field to 16.125 kHz and preserves unavailable calibrated power.

## Backend and dashboard

The RF-row consumer is `src/backend/auth_engine.py`. The five demonstration corridors come from `src/backend/rules.py` by default. This handoff did not contain a separate Laptop 1 rule JSON, master orchestrator, or serialized team model. The backend can read a team configuration from `config/disasternet_rules.json` when provided.

For the dashboard use `src/dashboard/START-HERE.md`. Its local dependencies include PyArrow 17.0.0 for the team's NumPy pin. To exercise both suites in PowerShell with the pinned dashboard dependencies installed:

    $env:KAAVALU_REPLAY_DIR = (Get-Location).Path
    python -m pytest -q tests

The master handover and Part 2 RF/ML guide are under `docs/`. The original replay ZIP and combined integration handoff ZIP are included for direct download.
