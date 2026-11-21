"""Presentation evidence. The backend remains the authority for every verdict."""

import csv
from datetime import datetime, timezone
import io

from src.backend import check_authorization


def evaluate(records, rules):
    checked = check_authorization(records, rules)
    lookup = {rule["rule_id"]: rule for rule in rules}
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in checked:
        row["dashboard_rule_snapshot"] = lookup.get(row["matched_rule_id"])
        row["dashboard_evaluated_at_utc"] = when
    return checked


def incident(row, rules, history):
    rule_id = row["matched_rule_id"]
    if "dashboard_rule_snapshot" in row:
        rule = row["dashboard_rule_snapshot"]
        basis = "At evaluation" if rule else "No match"
    else:
        rule = next((r for r in rules if r["rule_id"] == rule_id), None)
        basis = "Current configuration only" if rule else ("Snapshot unavailable" if rule_id else "No match")
    width = row["occupied_bw_khz"]
    nominal = rule["max_bw_khz"] if rule else None
    # Mirrors the documented backend display tolerance; never determines the verdict.
    limit = round(nominal * 1.15, 10) if nominal is not None else None
    hypothetical = row.get("calibration_status") == "HYPOTHETICAL_RULE_INPUT"
    reviews = [r for r in history if r["session_id"] == row["session_id"] and r["signal_id"] == row["signal_id"]]
    missing = []
    if hypothetical or row["peak_power_dbm"] is None:
        missing.append("Calibrated power")
    if not row.get("capture_timestamp"):
        missing.append("Capture timestamp")
    for key, label in (("snr_db", "SNR"), ("classification_confidence", "Classifier confidence")):
        if row.get(key) is None:
            missing.append(label)
    return dict(session_id=row["session_id"], signal_id=row["signal_id"], source_tag=row["source_tag"],
                evidence_type=row.get("dashboard_evidence_type", "Measured RF"),
                capture_timestamp=row.get("capture_timestamp"),
                evaluated_at_utc=row.get("dashboard_evaluated_at_utc"), center_freq_mhz=row["center_freq_mhz"],
                occupied_bw_khz=width, snr_db=row.get("snr_db"),
                peak_power_dbm=None if hypothetical else row["peak_power_dbm"],
                assumed_power_dbm=row["peak_power_dbm"] if hypothetical else None,
                predicted_modulation=row.get("predicted_modulation"), classification_confidence=row.get("classification_confidence"),
                classifier_reason=row.get("clf_reason"), matched_rule_id=rule_id, agency_corridor=row["agency_corridor"],
                rule_basis=basis, nominal_limit_khz=nominal, adjusted_limit_khz=limit,
                excess_bw_khz=max(0, round(width-limit, 10)) if limit is not None else None,
                allowed_modulations=", ".join(rule["allowed_modulations"]) if rule else None,
                power_limit_dbm=rule.get("max_power_dbm") if rule else None,
                verdict_category=row["verdict_category"], alert_severity=row["alert_severity"],
                explainable_reason=row["explainable_reason"], missing_measurements=", ".join(missing),
                review_status=("Training preview" if row.get("dashboard_evidence_type") else
                               reviews[0]["disposition"] if reviews else "Unreviewed"))


def evidence_csv(rows, rules, history):
    """One row per decision, including incidents that have never been reviewed."""
    lines = []
    for row in rows:
        evidence = incident(row, rules, history)
        reviews = [r for r in history if r["session_id"] == row["session_id"] and r["signal_id"] == row["signal_id"]]
        for review in reviews or [{}]:
            lines.append(dict(evidence, review_id=review.get("review_id"), operator=review.get("coordinator_name"),
                              operator_role=review.get("coordinator_role"), decision=review.get("disposition"),
                              reviewed_at_utc=review.get("reviewed_at_utc"), notes=review.get("notes")))
    stream = io.StringIO(newline="")
    if lines:
        writer = csv.DictWriter(stream, fieldnames=list(lines[0]))
        writer.writeheader()
        for line in lines:
            # Protect spreadsheet readers while keeping numeric measurements numeric.
            writer.writerow({key: "'" + val if isinstance(val, str) and val.lstrip().startswith(("=", "+", "-", "@"))
                             else val for key, val in line.items()})
    return stream.getvalue()
