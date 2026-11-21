"""Local DisasterNet demonstration rules. Frequencies are in MHz, widths in kHz."""

import math
from numbers import Real


MODULATIONS = {"GFSK", "FM-NB", "BPSK", "QPSK", "AM-DSB", "WB-FM"}
SOURCES = {"SIMULATED", "RECORDED_IQ", "LIVE_SDR"}
TIERS = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def demo_rules():
    rows = [
        ("EM-POL-01", "Police Tactical & Traffic Dispatch", "Police Comms Coordinator",
         148, 148.15, ["GFSK", "FM-NB"], 25, -15, "CRITICAL"),
        ("EM-FIR-02", "Fire & Hazmat Tactical Net", "Fireground Frequency Coordinator",
         148.15, 148.30, ["FM-NB", "GFSK"], 25, -15, "CRITICAL"),
        ("EM-AMB-03", "Ambulance & EMS Medical Telemetry", "EMS Medical Dispatch Coordinator",
         148.30, 148.45, ["GFSK", "BPSK"], 35, -20, "CRITICAL"),
        ("EM-RES-04", "Disaster Search & Rescue (NDRF / SAR)", "Rescue Operations Coordinator",
         148.45, 148.65, ["QPSK", "GFSK"], 50, -10, "HIGH"),
        ("EM-AIR-05", "Relief Helicopter Air-to-Ground SAR", "Aviation SAR Coordinator",
         148.65, 148.85, ["AM-DSB"], 30, -15, "CRITICAL"),
    ]
    keys = ("rule_id", "agency_corridor", "coordinator_role", "freq_min_mhz", "freq_max_mhz",
            "allowed_modulations", "max_bw_khz", "max_power_dbm", "priority_tier")
    return [dict(zip(keys, row)) for row in rows]


def validated_rules(rules):
    result = []
    ids = set()
    for item in rules:
        rule = dict(item)
        for key in ("rule_id", "agency_corridor", "coordinator_role"):
            if not isinstance(rule.get(key), str) or not rule[key].strip():
                raise ValueError(f"Rule needs {key}")
        if rule["rule_id"] in ids:
            raise ValueError(f"Duplicate rule ID {rule['rule_id']}")
        ids.add(rule["rule_id"])
        for key in ("freq_min_mhz", "freq_max_mhz", "max_bw_khz"):
            if not finite(rule.get(key)) or rule[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if rule["freq_min_mhz"] >= rule["freq_max_mhz"]:
            raise ValueError("Each corridor needs a positive frequency span")
        if "max_power_dbm" not in rule or (
            rule["max_power_dbm"] is not None and not finite(rule["max_power_dbm"])
        ):
            raise ValueError("max_power_dbm must be a finite number or null")
        mods = rule.get("allowed_modulations")
        if not isinstance(mods, list) or not mods or any(m not in MODULATIONS for m in mods):
            raise ValueError("Select at least one known modulation for each rule")
        if rule.get("priority_tier") not in TIERS:
            raise ValueError("Rule has an invalid priority_tier")
        result.append(rule)
    result.sort(key=lambda rule: rule["freq_min_mhz"])
    for left, right in zip(result, result[1:]):
        if left["freq_max_mhz"] > right["freq_min_mhz"]:
            raise ValueError("Corridors overlap. Use distinct frequency ranges.")
    return result
