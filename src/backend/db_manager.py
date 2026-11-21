"""SQLite storage for configured rules, physical measurements, alerts and reviews."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from src.contracts import validate_measurements
from .alert_engine import generate_alerts
from .rules import SOURCES, demo_rules, validated_rules


def _open(path):
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db(db_path: str = "kaavalu.db", default_rules_json_path=None) -> None:
    with closing(_open(db_path)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("BEGIN IMMEDIATE")
        first = not db.execute("SELECT 1 FROM sqlite_master WHERE name='authorization_rules'").fetchone()
        db.execute("""CREATE TABLE IF NOT EXISTS authorization_rules (
            rule_id TEXT PRIMARY KEY, rule_json TEXT NOT NULL,
            corridor_name TEXT AS (json_extract(rule_json,'$.agency_corridor')),
            coordinator_role TEXT AS (json_extract(rule_json,'$.coordinator_role')),
            freq_min_hz REAL AS (json_extract(rule_json,'$.freq_min_mhz') * 1000000),
            freq_max_hz REAL AS (json_extract(rule_json,'$.freq_max_mhz') * 1000000),
            allowed_modulations TEXT AS (json_extract(rule_json,'$.allowed_modulations')),
            max_bandwidth_hz REAL AS (json_extract(rule_json,'$.max_bw_khz') * 1000),
            max_power_dbm REAL AS (json_extract(rule_json,'$.max_power_dbm')),
            priority_level TEXT AS (json_extract(rule_json,'$.priority_tier')))""")
        db.execute("""CREATE TABLE IF NOT EXISTS signal_detections (
            session_id TEXT NOT NULL, signal_id TEXT NOT NULL, source_tag TEXT NOT NULL,
            occupied_bw_khz REAL NOT NULL, peak_power_dbm REAL, verdict_category TEXT NOT NULL,
            record_json TEXT NOT NULL,
            capture_timestamp TEXT AS (json_extract(record_json,'$.capture_timestamp')),
            start_time_s REAL AS (json_extract(record_json,'$.start_time_s')),
            duration_s REAL AS (json_extract(record_json,'$.duration_s')),
            center_freq_hz REAL AS (json_extract(record_json,'$.center_freq_mhz') * 1000000),
            bandwidth_hz REAL AS (occupied_bw_khz * 1000), power_dbm REAL AS (peak_power_dbm),
            snr_db REAL AS (json_extract(record_json,'$.snr_db')),
            predicted_class TEXT AS (json_extract(record_json,'$.predicted_modulation')),
            confidence REAL AS (json_extract(record_json,'$.classification_confidence')),
            corridor_name TEXT AS (json_extract(record_json,'$.agency_corridor')),
            explanation TEXT AS (json_extract(record_json,'$.explainable_reason')),
            features_json TEXT AS (json_extract(record_json,'$.features_json')),
            PRIMARY KEY(session_id,signal_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS generated_alerts (
            alert_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, signal_id TEXT NOT NULL,
            severity TEXT NOT NULL, coordinator_role TEXT NOT NULL, corridor_name TEXT NOT NULL,
            alert_title TEXT NOT NULL, explainable_message TEXT NOT NULL, created_at_utc TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id,signal_id), FOREIGN KEY(session_id,signal_id) REFERENCES signal_detections)""")
        db.execute("""CREATE TABLE IF NOT EXISTS coordinator_reviews (
            review_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, signal_id TEXT NOT NULL,
            coordinator_name TEXT NOT NULL, coordinator_role TEXT NOT NULL, disposition TEXT NOT NULL,
            notes TEXT NOT NULL, reviewed_at_utc TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id,signal_id) REFERENCES signal_detections)""")
        if first:
            path = Path(default_rules_json_path or "config/disasternet_rules.json")
            if default_rules_json_path or path.exists():
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                entries = data["rules"] if isinstance(data, dict) else data
            else:
                entries = demo_rules()
            _write_rules(db, entries)


def _write_rules(db, entries):
    rules = validated_rules(entries)
    db.execute("DELETE FROM authorization_rules")
    db.executemany("INSERT INTO authorization_rules VALUES (?,?)",
                   [(r["rule_id"], json.dumps(r, allow_nan=False)) for r in rules])


def get_auth_rules(db_path: str = "kaavalu.db") -> list[dict]:
    with closing(_open(db_path)) as db:
        rows = db.execute("SELECT rule_json FROM authorization_rules")
        return validated_rules([json.loads(row[0]) for row in rows])


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported measurement type {type(value).__name__}")


def _save(db, rows):
    for row in rows:
        validate_measurements(row)
        if row["source_tag"] not in SOURCES:
            raise ValueError("Invalid source_tag")
        data = {k: v for k, v in row.items() if k not in {"iq_data", "baseband_iq", "iq_baseband"}}
        payload = json.dumps(data, allow_nan=False, default=_json_value)
        db.execute("""INSERT INTO signal_detections VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(session_id,signal_id) DO UPDATE SET source_tag=excluded.source_tag,
            occupied_bw_khz=excluded.occupied_bw_khz, peak_power_dbm=excluded.peak_power_dbm,
            verdict_category=excluded.verdict_category,record_json=excluded.record_json""",
                   (row["session_id"], row["signal_id"], row["source_tag"], row["occupied_bw_khz"],
                    row["peak_power_dbm"], row["verdict_category"], payload))
        alerts = generate_alerts([row], {})
        if alerts:
            db.execute("""INSERT INTO generated_alerts
                (alert_id,session_id,signal_id,severity,coordinator_role,corridor_name,alert_title,explainable_message)
                VALUES (:alert_id,:session_id,:signal_id,:severity,:coordinator_role,:corridor_name,:alert_title,:explainable_message)
                ON CONFLICT(alert_id) DO UPDATE SET severity=excluded.severity, coordinator_role=excluded.coordinator_role,
                corridor_name=excluded.corridor_name,alert_title=excluded.alert_title,explainable_message=excluded.explainable_message""",
                       alerts[0])
        else:
            db.execute("DELETE FROM generated_alerts WHERE session_id=? AND signal_id=?",
                       (row["session_id"], row["signal_id"]))


def save_detected_signals(signals_list: list[dict], db_path: str = "kaavalu.db") -> None:
    with closing(_open(db_path)) as db, db:
        _save(db, signals_list)


def save_rules_and_results(entries, records, db_path="kaavalu.db"):
    with closing(_open(db_path)) as db, db:
        _write_rules(db, entries)
        _save(db, records)


def list_sessions(db_path="kaavalu.db"):
    with closing(_open(db_path)) as db:
        return [r[0] for r in db.execute("SELECT session_id FROM signal_detections GROUP BY session_id ORDER BY max(rowid) DESC")]


def load_session(session_id, db_path="kaavalu.db"):
    with closing(_open(db_path)) as db:
        result = db.execute("SELECT record_json FROM signal_detections WHERE session_id=? ORDER BY rowid", (session_id,))
        return [json.loads(row[0]) for row in result]


def insert_coordinator_review(session_id, signal_id, name, role, disposition, notes, db_path="kaavalu.db"):
    if not all(s.strip() for s in (name, role, disposition)):
        raise ValueError("Enter a coordinator name, role and disposition")
    with closing(_open(db_path)) as db, db:
        db.execute("""INSERT INTO coordinator_reviews
            (session_id,signal_id,coordinator_name,coordinator_role,disposition,notes) VALUES (?,?,?,?,?,?)""",
                   (session_id, signal_id, name.strip(), role.strip(), disposition, notes.strip()))


def review_history(session_id, db_path="kaavalu.db"):
    with closing(_open(db_path)) as db:
        rows = db.execute("SELECT * FROM coordinator_reviews WHERE session_id=? ORDER BY review_id DESC", (session_id,))
        return [dict(r) for r in rows]
