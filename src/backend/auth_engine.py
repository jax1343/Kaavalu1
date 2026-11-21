"""Evaluate physical measurements against configured demonstration corridors."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import json

from src.contracts import to_shared_record
from .rules import MODULATIONS, SOURCES, finite, validated_rules


def _classified(row):
    data = dict(to_shared_record(row))
    # A digital level cannot supply an absolute dBm measurement. Normalize
    # explicit uncalibrated input before either policy or storage consumes it.
    if data.get("calibration_status") == "UNCALIBRATED":
        data["peak_power_dbm"] = None
    if data.get("source_tag") not in SOURCES:
        raise ValueError("Laptop 2 record needs a valid source_tag")
    if not data.get("signal_id"):
        raise ValueError("Laptop 2 record needs signal_id")
    data["session_id"] = data.get("session_id") or data.get("capture_id")
    if not data["session_id"]:
        raise ValueError("Laptop 2 record needs session_id or capture_id")
    mhz = data.get("center_freq_mhz")
    hz = data.get("center_freq_hz")
    if mhz is None and finite(hz):
        mhz = hz / 1e6
    if not finite(mhz) or mhz <= 0:
        raise ValueError("Laptop 2 record needs a finite, positive center frequency")
    if hz is not None and (not finite(hz) or abs(hz - mhz * 1e6) > 0.01):
        raise ValueError("center_freq_hz and center_freq_mhz disagree")
    data["center_freq_mhz"] = mhz
    data.setdefault("start_time_s", data.get("start_s"))
    return data


def _match(freq, rules):
    # At a shared edge the corridor starting at that frequency owns the point.
    for rule in reversed(rules):
        if rule["freq_min_mhz"] <= freq <= rule["freq_max_mhz"]:
            return rule
    return None


def _decision(row, rule, threshold):
    mod = row.get("predicted_modulation")
    snr = row.get("snr_db")
    conf = row.get("classification_confidence")
    if not finite(snr) or snr < 8:
        return "UNKNOWN", "SNR is unavailable or below 8 dB. Human review required."
    if row.get("is_unknown") is not False or mod not in MODULATIONS:
        return "UNKNOWN", "The classifier could not identify this signal reliably. Human review required."
    if not finite(conf) or not threshold <= conf <= 1:
        return "UNKNOWN", "Classification confidence is missing or outside the accepted range. Human review required."
    if rule is None:
        return "RECOGNIZED_NO_AUTH_RECORD", f"{mod} recognized outside the configured demonstration corridors."
    power = row["peak_power_dbm"]
    if rule["max_power_dbm"] is not None and power is None:
        return "UNKNOWN", "Power unavailable. This corridor requires calibrated power. Human review required."
    failures = []
    if mod not in rule["allowed_modulations"]:
        failures.append(f"{mod} is outside the permitted modulation list")
    bw_limit = round(rule["max_bw_khz"] * 1.15, 10)
    if row["occupied_bw_khz"] > bw_limit:
        failures.append(f"bandwidth {row['occupied_bw_khz']:g} kHz exceeds {bw_limit:g} kHz including tolerance")
    if rule["max_power_dbm"] is not None and power > rule["max_power_dbm"] + 2:
        failures.append(f"power {power:g} dBm exceeds {rule['max_power_dbm'] + 2:g} dBm including tolerance")
    if failures:
        return "POTENTIALLY_UNAUTHORIZED", ". ".join(failures) + ". Review against the demonstration rule."
    return "KNOWN_AUTHORIZED", "Measurements match the configured demonstration rule."


def check_authorization(classified_results: list[dict], rules_list: list[dict]) -> list[dict]:
    rules = validated_rules(rules_list)
    rows = [_classified(row) for row in classified_results]
    frequencies = defaultdict(list)
    for row in rows:
        frequencies[row["session_id"]].append(row["center_freq_mhz"] * 1e6)
    for values in frequencies.values():
        values.sort()
    for row in rows:
        freq = row["center_freq_mhz"]
        group = frequencies[row["session_id"]]
        count = bisect_right(group, freq * 1e6 + 20000) - bisect_left(group, freq * 1e6 - 20000)
        rule = _match(freq, rules)
        verdict, reason = _decision(row, rule, 0.68)
        if verdict == "POTENTIALLY_UNAUTHORIZED":
            severity = "CRITICAL"
        elif verdict == "UNKNOWN" and count >= 2:
            severity = "HIGH"
        elif verdict == "KNOWN_AUTHORIZED":
            severity = "INFO"
        else:
            severity = "MEDIUM"
        row.update(verdict_category=verdict, alert_severity=severity, recurrence_count=count,
                   unusual_activity_flag="REVIEW" if severity != "INFO" else "CLEAR",
                   agency_corridor=rule["agency_corridor"] if rule else "Unmapped",
                   matched_rule_id=rule["rule_id"] if rule else None,
                   coordinator_role=rule["coordinator_role"] if rule else "Spectrum coordinator",
                   explainable_reason=f"{freq:.6f} MHz · {count} burst(s) within 20 kHz. {reason}")
        for key, dest in (("features", "features_json"), ("psd_freqs_khz", "psd_freqs_json"),
                          ("psd_power_db", "psd_power_json"), ("constellation_i", "constellation_i_json"),
                          ("constellation_q", "constellation_q_json")):
            value = row.get(key)
            if hasattr(value, "tolist"):
                value = value.tolist()
            row[dest] = json.dumps(value, allow_nan=False)
    return rows
