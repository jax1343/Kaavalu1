# Laptop 3 local dashboard

Extract the Laptop 3 ZIP into the KAAVALU project folder containing Laptop 1's shared `src/contracts.py`. This version uses `to_shared_record`, `validate_measurements` and `power_display_text` from the shared file.

From that folder, with the project's Python 3.11 environment active, run:

```powershell
python -m pip install -r src/dashboard/requirements-local.txt
python -m streamlit run src/dashboard/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

In the sidebar, select the Laptop 2 replay ZIP and click **Load saved replay**. If the original ZIP is in the current folder, the button can load it directly. This view reads the saved detector record and spectrum. The integration test below runs the detector again from the IQ fixture.

The `familiar_clear` capture should show **SIMULATED**, **QPSK**, **16.125 kHz**, and **Power unavailable.** Its center frequency is outside the five demo corridors, so the expected verdict is **RECOGNIZED_NO_AUTH_RECORD**. To demonstrate missing-power review, add a temporary rule spanning its displayed frequency with a power limit and select **Save & Re-Evaluate**. The selected session updates to **UNKNOWN**.

The app stores data in `kaavalu.db` in the current folder. Set `KAAVALU_DB_PATH` to use another database. On first use it loads `config/disasternet_rules.json` if present or seeds the five handover demonstration rules. Shared frequency boundaries belong to the corridor starting at that frequency. The capture spectrum and waterfall are available after loading the packet. Stored sessions retain per-signal measurements, constellation, alerts and reviews.

For the RF-to-dashboard test, extract the original Laptop 2 ZIP. Set the path in PowerShell and run:

```powershell
$env:KAAVALU_REPLAY_DIR = (Resolve-Path "$env:USERPROFILE/Downloads/KAAVALU-Laptop-2-Replay").Path
python -m pytest -q tests/test_backend_auth.py
```

Laptop 1 can import the five backend functions from `src.backend` and `render_dashboard` from `src.dashboard`. Call `init_db()` once before reading rules or saving detections. Pass the pipeline function to `render_dashboard(pipeline_callback)`. The callback receives the scenario or recording path and current rules, and returns records plus display metadata. `check_authorization` produces enriched records for storage. `generate_alerts` returns only actionable alerts. Saving detections also persists their alerts in the same transaction.
