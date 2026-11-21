"""DisasterNet's offline coordinator dashboard. The callback is owned by Laptop 1."""

import json
import os
from pathlib import Path
import sqlite3
from zipfile import BadZipFile, ZipFile

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.backend import check_authorization, get_auth_rules, init_db, save_detected_signals
from src.backend.db_manager import (insert_coordinator_review, list_sessions, load_session,
                                    review_history, save_rules_and_results)
from src.backend.rules import MODULATIONS, TIERS
from src.contracts import power_display_text


COLORS = {"UNKNOWN": "#bc8708", "KNOWN_AUTHORIZED": "#267652",
          "POTENTIALLY_UNAUTHORIZED": "#be4245", "RECOGNIZED_NO_AUTH_RECORD": "#54769a"}


def _spectrum(meta, rows, rules):
    freq = meta.get("spectrum_freq_hz")
    if freq is not None:
        axis = [f / 1e6 for f in freq]
        values = meta["spectrum_db"]
        unit = meta.get("spectrum_unit", "Relative dB")
    else:
        axis = meta.get("freq_axis_mhz", [])
        values = meta.get("peak_psd_dbm", [])
        unit = "dBm"
    if not len(axis):
        st.caption("Spectrum data is unavailable for this session.")
        return
    times = meta.get("time_s", meta.get("time_axis_s", []))
    waterfall = meta.get("waterfall_db", [])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.09)
    fig.add_trace(go.Scatter(x=axis, y=values, mode="lines", line=dict(color="#276278")), row=1, col=1)
    if len(times) and len(waterfall):
        water_axis = [f / 1e6 for f in meta["frequency_hz"]] if "frequency_hz" in meta else axis
        fig.add_trace(go.Heatmap(x=water_axis, y=times, z=waterfall, colorscale="Viridis",
                                 showscale=False), row=2, col=1)
        for row in rows:
            start, duration = row.get("start_time_s"), row.get("duration_s")
            if start is not None and duration is not None:
                center, half = row["center_freq_mhz"], row["occupied_bw_khz"] / 2000
                fig.add_shape(type="rect", x0=center-half, x1=center+half, y0=start, y1=start+duration,
                              line=dict(color=COLORS[row["verdict_category"]], width=2), row=2, col=1)
    for rule in rules:
        lo, hi = max(rule["freq_min_mhz"], min(axis)), min(rule["freq_max_mhz"], max(axis))
        if lo < hi:
            fig.add_vrect(x0=lo, x1=hi, fillcolor="#7eb6a3", opacity=.2, line_width=0, row=1, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (MHz)", row=2, col=1)
    fig.update_layout(height=450, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _evidence(rows, meta, rules):
    _spectrum(meta, rows, rules)
    if not rows:
        st.info("No detections to inspect.")
        return
    table = [{"Signal": r["signal_id"], "Source": r["source_tag"], "MHz": r["center_freq_mhz"],
              "Bandwidth (kHz)": r["occupied_bw_khz"], "Power": power_display_text(r["peak_power_dbm"]),
              "SNR (dB)": r.get("snr_db"), "Modulation": r.get("predicted_modulation"),
              "Verdict": r["verdict_category"], "Severity": r["alert_severity"]} for r in rows]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    by_id = {r["signal_id"]: r for r in rows}
    row = by_id[st.selectbox("Inspect signal", list(by_id))]
    columns = st.columns(3)
    columns[0].metric("Occupied bandwidth", f"{row['occupied_bw_khz']:g} kHz")
    columns[1].metric("Peak power", power_display_text(row["peak_power_dbm"]))
    columns[2].metric("Severity", row["alert_severity"])
    st.write(row["explainable_reason"])
    st.caption(f"{row['agency_corridor']} · {row['coordinator_role']}")
    i, q = row.get("constellation_i"), row.get("constellation_q")
    if i is not None and len(i) and q is not None and len(i) == len(q):
        fig = go.Figure(go.Scattergl(x=i, y=q, mode="markers", marker=dict(size=4, color="#217978")))
        fig.update_layout(title="IQ constellation", xaxis_title="In-phase", yaxis_title="Quadrature",
                          height=280, margin=dict(t=45, b=30))
        st.plotly_chart(fig, use_container_width=True)
    if row.get("features"):
        with st.expander("Waveform features"):
            st.json(row["features"])


def _edit_rules(rules, rows, db_path):
    st.caption("Demonstration corridors. Bandwidth includes a 15% tolerance and power a 2 dB tolerance.")
    keys = ["rule_id", "agency_corridor", "coordinator_role", "freq_min_mhz", "freq_max_mhz",
            "allowed_modulations", "max_bw_khz", "max_power_dbm", "priority_tier"]
    frame = pd.DataFrame(rules, columns=keys)
    frame["allowed_modulations"] = frame["allowed_modulations"].map(lambda mods: ", ".join(mods))
    edited = st.data_editor(frame, hide_index=True, use_container_width=True, num_rows="dynamic",
                            key="rule_editor", column_config={
                                "priority_tier": st.column_config.SelectboxColumn(options=sorted(TIERS)),
                                "allowed_modulations": st.column_config.TextColumn(help=", ".join(sorted(MODULATIONS)))})
    if st.button("Save & Re-Evaluate"):
        entries = edited.astype(object).where(edited.notna(), None).to_dict("records")
        try:
            for entry in entries:
                entry["allowed_modulations"] = [s.strip() for s in (entry["allowed_modulations"] or "").split(",")]
            checked = check_authorization(rows, entries)
            save_rules_and_results(entries, checked, db_path)
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _reviews(rows, session, db_path):
    if not rows:
        st.info("Select a session to review its detections.")
        return
    flagged = [r["signal_id"] for r in rows if r["verdict_category"] != "KNOWN_AUTHORIZED"]
    if flagged:
        with st.form("coordinator_review", clear_on_submit=True):
            signal = st.selectbox("Flagged signal", flagged)
            name, role = st.columns(2)
            coordinator = name.text_input("Coordinator name")
            agency = role.text_input("Coordinator role")
            disposition = st.selectbox("Disposition", ["Needs follow-up", "Verified friendly", "Confirmed spillover"])
            notes = st.text_area("Investigation notes")
            if st.form_submit_button("Save review"):
                try:
                    insert_coordinator_review(session, signal, coordinator, agency, disposition, notes, db_path)
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
                else:
                    st.success("Review saved.")
    history = review_history(session, db_path)
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
    report = {"session_id": session, "detections": rows, "reviews": history}
    st.download_button("Download JSON report", json.dumps(report, indent=2), "session.json", "application/json")
    summary = pd.DataFrame(rows)[["source_tag", "session_id", "signal_id", "center_freq_mhz", "occupied_bw_khz",
                                  "peak_power_dbm", "verdict_category", "alert_severity", "explainable_reason"]]
    st.download_button("Download detections CSV", summary.to_csv(index=False), "detections.csv", "text/csv")


def _read_packet(packet):
    with ZipFile(packet) as archive:
        result = []
        for name in ("familiar_clear_outgoing.json", "familiar_clear_display.json"):
            member = archive.getinfo("handoff/" + name)
            if member.file_size > 16 * 1024 * 1024:
                raise ValueError("Replay JSON exceeds the display size limit")
            result.append(json.loads(archive.read(member)))
    row, display = result
    if (row["capture_id"], row["source_tag"]) != (display["capture_id"], display["source_tag"]):
        raise ValueError("The replay record and spectrum have different capture IDs or sources")
    return [row], display


def render_dashboard(pipeline_callback) -> None:
    db_path = os.environ.get("KAAVALU_DB_PATH", "kaavalu.db")
    init_db(db_path)
    rules = get_auth_rules(db_path)
    st.title("DisasterNet")
    st.caption("Offline spectrum coordination · Physical signal measurements only · Unknown signals require human review.")
    st.sidebar.header("Capture")
    if pipeline_callback is None:
        upload = st.sidebar.file_uploader("Laptop 2 replay ZIP", type="zip")
        default = Path("KAAVALU-Laptop-2-Replay.zip")
        packet = upload if upload is not None else default if default.is_file() else None
        run = st.sidebar.button("Load saved replay", disabled=packet is None)
    else:
        source = st.sidebar.text_input("Scenario or recording path", "familiar_clear")
        run = st.sidebar.button("Run analysis")
    if run:
        try:
            if pipeline_callback is None:
                records, metadata = _read_packet(packet)
            else:
                records, metadata = pipeline_callback(source, rules)
            rows = check_authorization(records, rules)
            save_detected_signals(rows, db_path)
        except (ValueError, OSError, KeyError, BadZipFile, sqlite3.Error) as exc:
            st.error(f"Capture could not be loaded: {exc}")
            return
        if rows:
            st.session_state["session_picker"] = rows[0]["session_id"]
        st.session_state["active_display"] = (rows[0]["session_id"] if rows else None, metadata)
        if not rows:
            st.info(f"{metadata.get('source_tag', 'Source unavailable')} · No detections in this capture.")
            _spectrum(metadata, [], rules)
            return
    sessions = list_sessions(db_path)
    session = st.sidebar.selectbox("Stored session", sessions, key="session_picker") if sessions else None
    rows = load_session(session, db_path) if session else []
    active, meta = st.session_state.get("active_display", (None, {}))
    meta = meta if active == session else {}
    source_label = ", ".join(sorted({r["source_tag"] for r in rows})) or meta.get("source_tag", "No capture selected")
    st.subheader(source_label)
    evidence, policies, review = st.tabs(["Signal evidence", "Demonstration rules", "Coordinator review"])
    with evidence:
        _evidence(rows, meta, rules)
    with policies:
        _edit_rules(rules, rows, db_path)
    with review:
        _reviews(rows, session, db_path)
