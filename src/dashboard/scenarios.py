"""Adapters for the dashboard's named inputs and explicitly isolated policy previews."""

FAMILIAR = "Familiar QPSK"
AUTHORIZED = "Authorized signal · demo rule"
FIRE = "Fire channel · power unavailable"
UNKNOWN = "Unknown modulation"
HYPOTHETICAL = "Fire conflict · rule-only"
PACKET = "Saved replay ZIP"
CUSTOM = "Scenario or recording path"
SCENARIOS = {FAMILIAR: "familiar_clear", AUTHORIZED: "familiar_clear", FIRE: "fire_wide",
             UNKNOWN: "unfamiliar_clear", HYPOTHETICAL: "fire_wide"}


def builtin_input(source):
    # Saved sessions and ZIP replay also work with a standalone dashboard install.
    if source == "familiar_clear":
        from replay_familiar_clear import replay
        row, display = replay()
        return [row], display
    from src.rf_processing import classify_signals, extract_features, generate_simulated_disasternet_capture
    if source == "fire_wide":
        from src.rf_processing.dataset_generator import simulate_recording
        from src.rf_processing.iq_loader import analyze_recording
        recording = simulate_recording("WB-FM", 61080, tuning_hz=148225000.0, offset_hz=0.0)
        bursts, display = analyze_recording(recording)
    else:
        bursts, display = generate_simulated_disasternet_capture(source)
    return classify_signals(extract_features(bursts, display["sample_rate_hz"])), display


def prepare_case(choice, records, rules):
    if choice == AUTHORIZED:
        demo = dict(rule_id="DEMO-QPSK", agency_corridor="QPSK training corridor",
                    coordinator_role="Training coordinator", freq_min_mhz=150.0, freq_max_mhz=150.05,
                    allowed_modulations=["QPSK"], max_bw_khz=25, max_power_dbm=None, priority_tier="LOW")
        return [dict(r, dashboard_evidence_type="Measured RF / demo-rule preview") for r in records], [demo]
    if choice == HYPOTHETICAL:
        return [dict(r, signal_id=r["signal_id"] + "/rule-only", peak_power_dbm=-20.0,
                     calibration_status="HYPOTHETICAL_RULE_INPUT", dashboard_evidence_type="Hypothetical rule-only")
                for r in records], rules
    return records, rules
