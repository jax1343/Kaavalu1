"""DisasterNet's offline coordinator dashboard. The callback is owned by Laptop 1."""

import json
import os
from pathlib import Path
import sqlite3
from zipfile import BadZipFile, ZipFile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backend import get_auth_rules, init_db, save_detected_signals
from src.backend.db_manager import (insert_coordinator_review, list_sessions, load_session,
                                    review_history, save_rules_and_results)
from src.backend.rules import MODULATIONS, TIERS
from src.contracts import power_display_text


from .incidents import evaluate, evidence_csv, incident
from .monitor import COLORS, spectrum
from .scenarios import (AUTHORIZED, CUSTOM, HYPOTHETICAL, PACKET, SCENARIOS,
                        builtin_input, prepare_case)


def _width(width):
    return f"{width:.2f} kHz" if round(width, 2) == width else f"{width:g} kHz"


def _evidence(rows, meta, rules, history):
    by_id = {r["signal_id"]: r for r in rows}
    selected = st.selectbox("Inspect signal", list(by_id))
    row = by_id[selected]
    ev = incident(row, rules, history)
    verdict = ev["verdict_category"]
    st.markdown(f'<div class="verdict" style="border-color:{COLORS[verdict]}">'
                f'<small>Final verdict</small><strong>{verdict}</strong></div>', unsafe_allow_html=True)
    st.caption(f'{ev["agency_corridor"]} · {ev["alert_severity"]} severity · Review: {ev["review_status"]}')
    st.write(ev["explainable_reason"])
    cols = st.columns(3)
    cols[0].metric("Measured width", _width(ev["occupied_bw_khz"]))
    cols[1].metric("Adjusted limit", _width(ev["adjusted_limit_khz"]) if ev["adjusted_limit_khz"] is not None else "Unavailable")
    cols[2].metric("Excess bandwidth", _width(ev["excess_bw_khz"]) if ev["excess_bw_khz"] is not None else "Unavailable")
    if ev["nominal_limit_khz"] is not None:
        st.caption(f'{ev["rule_basis"]} · {ev["matched_rule_id"]} · '
                   f'{ev["nominal_limit_khz"]:g} kHz nominal limit with 15% tolerance.')
    cols = st.columns(3)
    cols[0].metric("Modulation", ev["predicted_modulation"] or "Unavailable")
    cols[1].metric("SNR", f'{ev["snr_db"]:.2f} dB' if ev["snr_db"] is not None else "Unavailable")
    cols[2].metric("Calibrated peak power", power_display_text(ev["peak_power_dbm"]))
    if ev["peak_power_dbm"] is None and "calibrated power" not in ev["explainable_reason"].lower():
        st.caption("Calibrated power is unavailable for this signal.")
    if ev["assumed_power_dbm"] is not None:
        st.warning(f'Rule-only input assumes {ev["assumed_power_dbm"]:g} dBm. The RF measurement has no calibrated power.')
    details = [
        ("Measured center", f'{ev["center_freq_mhz"]:.6f} MHz'),
        ("Capture timestamp", ev["capture_timestamp"] or "Unavailable"),
        ("Evaluated (UTC)", ev["evaluated_at_utc"] or "Unavailable"),
        ("Authorization lookup", ev["matched_rule_id"] or "No matched rule"),
        ("Allowed modulation", ev["allowed_modulations"] or "Unavailable"),
        ("Classification evidence", ev["classifier_reason"] or "Unavailable"),
        ("Missing evidence", ev["missing_measurements"] or "None recorded"),
    ]
    with st.expander("Alert details · frequency, timestamps and rule lookup"):
        st.table(pd.DataFrame(details, columns=["Field", "Evidence"]).set_index("Field"))
    st.markdown("#### Spectrum monitor")
    figure = spectrum(meta, rows, row, rules)
    if figure is None:
        st.info("Spectrum and time-series samples are unavailable for this stored session. Run the capture to inspect them.")
    else:
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Shading marks configured corridors and detected bands requiring review. "
                   "Time-series strength uses the peak measured waterfall bin inside the selected bandwidth.")
    i, q = row.get("constellation_i"), row.get("constellation_q")
    if i is not None and len(i) and q is not None and len(i) == len(q):
        fig = go.Figure(go.Scattergl(x=i, y=q, mode="markers", marker=dict(size=4, color="#207581")))
        fig.update_layout(title="IQ constellation", xaxis_title="In-phase", yaxis_title="Quadrature",
                          height=270, margin=dict(t=45,b=30))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Session measurements and waveform features"):
        st.dataframe([{"Signal": r["signal_id"], "Frequency (MHz)": r["center_freq_mhz"],
                       "Bandwidth (kHz)": r["occupied_bw_khz"], "Modulation": r.get("predicted_modulation"),
                       "Final verdict": r["verdict_category"]} for r in rows], hide_index=True)
        if row.get("features"):
            st.json(row["features"])
    return row


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
            checked = evaluate(rows, entries)
            save_rules_and_results(entries, checked, db_path)
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))
        else:
            st.rerun()


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


def _reviews(rows, selected, rules, history, db_path, preview):
    if preview:
        st.info("Training preview. Operator decisions are available for stored incidents.")
    else:
        with st.form("coordinator_review", clear_on_submit=True):
            st.markdown("#### Record a decision")
            st.caption(selected["signal_id"])
            name, role = st.columns(2)
            operator = name.text_input("Coordinator name")
            agency = role.text_input("Coordinator role")
            decision = st.radio("Disposition", ["Verified", "False alarm", "Escalate"], horizontal=True)
            notes = st.text_area("Investigation notes")
            if st.form_submit_button("Save review"):
                try:
                    insert_coordinator_review(selected["session_id"], selected["signal_id"], operator,
                                              agency, decision, notes, db_path)
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["review_notice"] = "Review saved."
                    st.rerun()
        st.markdown("#### Incident history")
        scope = st.radio("History scope", ["Selected session", "All reviewed sessions"], horizontal=True)
        archive = history
        if scope == "All reviewed sessions":
            archive = sorted((entry for session_id in list_sessions(db_path)
                              for entry in review_history(session_id, db_path)),
                             key=lambda entry: entry["review_id"], reverse=True)
        if archive:
            fields = {"reviewed_at_utc": "Time (UTC)", "signal_id": "Signal ID", "disposition": "Decision",
                      "session_id": "Session", "coordinator_name": "Operator",
                      "coordinator_role": "Role", "notes": "Notes"}
            st.dataframe(pd.DataFrame(archive)[list(fields)].rename(columns=fields),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("No operator decisions recorded.")
    st.markdown("#### Export evidence")
    name = "training-preview" if preview else "session"
    st.download_button("Download incident evidence CSV", evidence_csv(rows, rules, history),
                       f"{name}-evidence.csv", "text/csv")
    report = {"session_id": selected["session_id"], "detections": rows, "reviews": history}
    st.download_button("Download JSON report", json.dumps(report, indent=2), f"{name}.json", "application/json")


def _clear_preview():
    st.session_state.pop("case_preview", None)


def render_dashboard(pipeline_callback) -> None:
    db_path = os.environ.get("KAAVALU_DB_PATH", "kaavalu.db")
    init_db(db_path)
    rules = get_auth_rules(db_path)
    st.markdown("""<style>
      .main .block-container {max-width:82rem; padding-top:3.4rem;}
      [data-testid="stSidebar"] {border-right:1px solid #aab7bd55;}
      h1,h2,h3,h4 {letter-spacing:-.025em;}
      .verdict {border-left:5px solid;padding:.55rem 1rem;margin:.5rem 0;
                background:rgba(109,133,148,.09);border-radius:3px;}
      .verdict small,.verdict strong {display:block;}
      .verdict strong {font:600 clamp(1rem,3vw,1.35rem) ui-monospace,monospace;overflow-wrap:anywhere;}
      [data-testid="stMetric"] {border:1px solid #74889244;border-radius:5px;padding:.6rem .8rem;}
      [data-testid="stMetricValue"] {font-size:clamp(1.05rem,2vw,1.8rem);font-variant-numeric:tabular-nums;}
      [data-testid="stSidebar"] button[kind="primary"] {background:#176579;border-color:#176579;color:white;}
      @media(max-width:650px){.main .block-container{padding:3.2rem 1rem 1rem;}
        [data-baseweb="tab-list"]{flex-wrap:wrap;height:auto;gap:.25rem .75rem;}
        [data-baseweb="tab"]{padding:.3rem 0;font-size:.85rem;}}
    </style>""", unsafe_allow_html=True)
    st.title("KAAVALU · Incident monitor")
    st.caption("DisasterNet  /  Offline spectrum coordination")
    if "review_notice" in st.session_state:
        st.success(st.session_state.pop("review_notice"))
    st.sidebar.header("Capture controls")
    choices = list(SCENARIOS) + ([CUSTOM] if pipeline_callback is not None else []) + [PACKET]
    choice = st.sidebar.radio("Input scenario", choices)
    source = SCENARIOS.get(choice)
    packet = None
    if choice == CUSTOM:
        source = st.sidebar.text_input(CUSTOM, "familiar_clear")
    if choice == PACKET:
        upload = st.sidebar.file_uploader("Laptop 2 replay ZIP", type="zip")
        default = Path("KAAVALU-Laptop-2-Replay.zip")
        packet = upload if upload is not None else default if default.is_file() else None
    if choice == AUTHORIZED:
        st.sidebar.caption("Measured QPSK evaluated against an isolated training rule with no power requirement.")
    if choice == HYPOTHETICAL:
        st.sidebar.caption("Prepared policy example with an assumed −20 dBm input.")
    run = st.sidebar.button("Load saved replay" if choice == PACKET else "Run analysis",
                            disabled=choice == PACKET and packet is None, type="primary")
    if run:
        try:
            if choice == PACKET:
                records, metadata = _read_packet(packet)
            elif pipeline_callback is None:
                records, metadata = builtin_input(source)
            else:
                records, metadata = pipeline_callback(source, rules)
            records, evaluation_rules = prepare_case(choice, records, rules)
            checked = evaluate(records, evaluation_rules)
            if choice in (AUTHORIZED, HYPOTHETICAL):
                st.session_state["case_preview"] = (checked, metadata, evaluation_rules, choice)
            else:
                _clear_preview()
                save_detected_signals(checked, db_path)
                session = checked[0]["session_id"] if checked else None
                st.session_state["active_display"] = (session, metadata)
                if session:
                    st.session_state["session_picker"] = session
                else:
                    st.subheader(metadata.get("source_tag", "Source unavailable"))
                    st.info("No signals detected in this capture.")
                    fig = spectrum(metadata, [], None, rules)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    return
        except (ModuleNotFoundError, ValueError, OSError, KeyError, BadZipFile, sqlite3.Error) as exc:
            st.error(f"Capture could not be loaded: {exc}")
            return
    sessions = list_sessions(db_path)
    session = st.sidebar.selectbox("Stored session", sessions, key="session_picker",
                                   on_change=_clear_preview) if sessions else None
    preview = st.session_state.get("case_preview")
    if preview:
        rows, meta, active_rules, label = preview
        history = []
        st.info(f"Training preview · {label}. Excluded from stored incidents.")
        st.button("Return to stored incidents", on_click=_clear_preview)
    else:
        rows = load_session(session, db_path) if session else []
        active, meta = st.session_state.get("active_display", (None, {}))
        meta = meta if active == session else {}
        active_rules = rules
        history = review_history(session, db_path) if session else []
    sources = sorted({r["source_tag"] for r in rows})
    st.subheader(" / ".join(sources) if sources else "No capture selected")
    evidence, review, policies = st.tabs(["Signal evidence", "Incident review & export", "Demonstration rules"])
    with evidence:
        if rows:
            selected = _evidence(rows, meta, active_rules, history)
        else:
            st.info("Select an input in Capture controls and run analysis.")
    with review:
        if rows:
            _reviews(rows, selected, active_rules, history, db_path, bool(preview))
        else:
            st.caption("Load a session to review or export evidence.")
    with policies:
        if preview:
            st.caption("Return to stored incidents to edit the coordinator's rules.")
        else:
            _edit_rules(rules, rows, db_path)
