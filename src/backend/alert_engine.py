import json
from uuid import NAMESPACE_URL, uuid5


def generate_alerts(checked_results: list[dict], session_metadata: dict) -> list[dict]:
    alerts = []
    for row in checked_results:
        verdict = row["verdict_category"]
        if verdict == "KNOWN_AUTHORIZED":
            continue
        session = row.get("session_id") or session_metadata["session_id"]
        identity = json.dumps([session, row["signal_id"]])
        alerts.append(dict(alert_id=str(uuid5(NAMESPACE_URL, identity)), session_id=session,
                           signal_id=row["signal_id"], source_tag=row["source_tag"],
                           severity=row["alert_severity"], coordinator_role=row["coordinator_role"],
                           corridor_name=row["agency_corridor"], alert_title=verdict,
                           explainable_message=row["explainable_reason"]))
    return alerts
