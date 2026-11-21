# Laptop 3 local dashboard

Start in the Kaavalu1 checkout on `feature/dashboard-incident-management`. The standalone app uses Laptop 1's shared contract, Laptop 2's RF package and Laptop 3's existing SQLite and policy APIs.

From that folder, with the project's Python 3.11 environment active, run:

```powershell
python -m pip install -r src/dashboard/requirements-local.txt
python -m streamlit run src/dashboard/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

In **Capture controls**, select a scenario and click **Run analysis**. The familiar, fire and unknown scenarios execute the existing RF pipeline. To use the earlier packet workflow, select **Saved replay ZIP** and click **Load saved replay**. If the original ZIP is in the current folder, the button can load it directly.

| Scenario | Input and expected result with the default rules |
|---|---|
| Familiar QPSK | Bundled `fixtures/familiar_clear.npz`, QPSK at 16.125 kHz. Its center frequency falls outside the configured corridors. |
| Authorized signal · demo rule | The same actual capture compared with an isolated QPSK training rule covering 150.000–150.050 MHz, with a 25 kHz width limit and no power check. Expected preview verdict is `KNOWN_AUTHORIZED`. |
| Fire channel · power unavailable | Published WB-FM simulation recipe, seed 61080. The measured 33.00 kHz exceeds the adjusted fire limit of 28.75 kHz by 4.25 kHz. The required calibrated power is unavailable. Final verdict is `UNKNOWN`. |
| Unknown modulation | Existing `unfamiliar_clear` RF scenario. Inspect the classifier's reason and human-review verdict. |
| Fire conflict · rule-only | A copy of the measured fire record uses a separately labeled hypothetical −20 dBm rule input. With the default rules the backend returns `POTENTIALLY_UNAUTHORIZED`. |

The two previews stay in browser session memory and can be exported. They do not add incidents, reviews or rules to SQLite. Their verdicts are computed by the existing backend. All five scenarios use simulated evidence.

In **Signal evidence**, select the signal, inspect the verdict and expand the alert details for timestamps, rule lookup and missing measurements. The monitor shows measured spectrum, waterfall and peak spectral density over time within the selected occupied band. The latter uses the maximum measured waterfall bin for each time slice. Relative digital spectral density keeps its source units. Historical sessions without the display arrays show an availability message.

In **Incident review & export**, enter your name and role and select **Verified**, **False alarm** or **Escalate**. The latest decision appears beside the verdict. Earlier decisions, including legacy dispositions, remain in history. Select **All reviewed sessions** to browse across captures. CSV export includes the selected session's current incident evidence plus one line per linked review. Unreviewed signals have a line with empty review fields. JSON retains the full session records and reviews.

The app stores data in `kaavalu.db` in the current folder. Set `KAAVALU_DB_PATH` to use another database. On first use it loads `config/disasternet_rules.json` if present or seeds the five handover demonstration rules. Shared frequency boundaries belong to the corridor starting at that frequency. The capture spectrum and waterfall are available after loading the packet. Stored sessions retain per-signal measurements, constellation, alerts and reviews.

For the RF-to-dashboard test, extract the original Laptop 2 ZIP. Set the path in PowerShell and run:

```powershell
$env:KAAVALU_REPLAY_DIR = (Resolve-Path "$env:USERPROFILE/Downloads/KAAVALU-Laptop-2-Replay").Path
python -m pytest -q tests/test_backend_auth.py
```

Laptop 1 can import the five backend functions from `src.backend` and `render_dashboard` from `src.dashboard`. Call `init_db()` once before reading rules or saving detections. Pass the pipeline function to `render_dashboard(pipeline_callback)`. The callback receives the scenario or recording path and current rules, and returns records plus display metadata. `check_authorization` produces enriched records for storage. `generate_alerts` returns only actionable alerts. Saving detections also persists their alerts in the same transaction.

## Integration contract for the RF and backend developer

- `render_dashboard(None)` uses local RF adapters. A supplied callback receives `(source: str, rules: list[dict])` and returns `(classified_rows, display_metadata)`. It must return measurements before storage. Source tokens are `familiar_clear`, `fire_wide` and `unfamiliar_clear`; custom recording paths pass through unchanged. Policy previews reuse their base capture.
- RF rows retain their original identifiers, source and classifier fields. The existing contract accepts native Hz or canonical kHz. Calibrated power uses a nullable number. Missing capture timestamps stay empty. The dashboard stamps policy evaluation time in UTC separately.
- The backend owns rule matching, guard bands and verdicts. The dashboard reports the backend verdict verbatim. Its displayed adjusted width follows the backend's 15% tolerance, and excess is the positive difference between occupied width and adjusted limit. Update both the display contract and tests if this policy changes.
- Dashboard-evaluated rows add `dashboard_rule_snapshot` and `dashboard_evaluated_at_utc` in the existing `record_json` payload. Previously stored rows without a rule snapshot explicitly label any corresponding rule as current configuration. No schema migration is required.
- Display metadata uses `spectrum_freq_hz`, `spectrum_db`, `spectrum_unit`, `frequency_hz`, `time_s`, and a time-by-frequency `waterfall_db`. The existing alternate MHz spectrum interface remains supported. Absent time samples produce no strength trace.
- Operator decisions use the existing append-only `insert_coordinator_review` interface and its database-generated UTC timestamp. Exports distinguish captured power, hypothetical power, capture time, evaluation time and review time. Linked review history describes operator actions; the snapshot fields describe the displayed evaluation.

## Changed files and verification

`dashboard_ui.py` owns capture selection, alert display, history and exports. `incidents.py` projects backend results into evidence fields. `monitor.py` builds the Plotly monitoring figure. `scenarios.py` maps the UI to the existing RF inputs. `app.py` opens the standalone dashboard in wide layout. This feature uses the previously pinned dashboard dependencies.

Run `python -m pytest -q tests/test_dashboard_incidents.py` for the dashboard contract and scenario checks. Set `KAAVALU_REPLAY_DIR` to the checkout root before running `python -m pytest -q tests` to include the bundled RF replay test.
