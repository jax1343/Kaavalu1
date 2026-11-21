"""Plot only the frequency, time and IQ samples actually supplied by the producer."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {"UNKNOWN": "#9a670c", "KNOWN_AUTHORIZED": "#187759",
          "POTENTIALLY_UNAUTHORIZED": "#bb3e35", "RECOGNIZED_NO_AUTH_RECORD": "#376a92"}


def spectrum(meta, rows, selected, rules):
    measured = "spectrum_freq_hz" in meta
    axis = np.asarray(meta.get("spectrum_freq_hz" if measured else "freq_axis_mhz", []), dtype=float)
    if measured:
        axis = axis / 1e6
    psd = meta.get("spectrum_db" if measured else "peak_psd_dbm", [])
    if not len(axis) or len(psd) != len(axis):
        return None
    unit = meta.get("spectrum_unit", "Relative dB" if measured else "dBm")
    times = np.asarray(meta.get("time_s", meta.get("time_axis_s", [])))
    water = np.asarray(meta.get("waterfall_db", []))
    freq = np.asarray(meta["frequency_hz"]) / 1e6 if "frequency_hz" in meta else axis
    has_water = bool(len(times) and len(freq) and water.shape == (len(times), len(freq)))
    lo = hi = None
    if selected:
        half = selected["occupied_bw_khz"] / 2000
        lo, hi = selected["center_freq_mhz"] - half, selected["center_freq_mhz"] + half
    bins = (freq >= lo) & (freq <= hi) if lo is not None else np.zeros(len(freq), dtype=bool)
    has_strength = has_water and bins.any()
    titles = ["Frequency spectrum"]
    if has_water:
        titles.append("Waterfall")
    if has_strength:
        titles.append("Peak spectral density in selected band")
    fig = make_subplots(rows=len(titles), cols=1, subplot_titles=titles, vertical_spacing=.13)
    fig.add_trace(go.Scatter(x=axis, y=psd, mode="lines", name="Spectrum",
                             line=dict(color="#176b83", width=1.6)), row=1, col=1)
    fig.update_xaxes(title_text="Frequency (MHz)", row=1, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    for rule in rules:
        left, right = max(rule["freq_min_mhz"], min(axis)), min(rule["freq_max_mhz"], max(axis))
        if left < right:
            fig.add_vrect(x0=left, x1=right, fillcolor="#8ebea9", opacity=.16, line_width=0, row=1, col=1)
    if has_water:
        fig.add_trace(go.Heatmap(x=freq, y=times, z=water, colorscale="Viridis",
                                 showscale=False, name="Waterfall"), row=2, col=1)
        fig.update_xaxes(title_text="Frequency (MHz)", matches="x", row=2, col=1)
        fig.update_yaxes(title_text="Time (s)", row=2, col=1)
    for r in rows:
        center, half = r["center_freq_mhz"], r["occupied_bw_khz"] / 2000
        color = COLORS[r["verdict_category"]]
        if r["verdict_category"] != "KNOWN_AUTHORIZED":
            fig.add_vrect(x0=center-half, x1=center+half, line_width=1, line_color=color,
                          fillcolor=color, opacity=.17, row=1, col=1)
        start, duration = r.get("start_time_s"), r.get("duration_s")
        if has_water and start is not None and duration is not None:
            fig.add_shape(type="rect", x0=center-half, x1=center+half, y0=start, y1=start+duration,
                          line=dict(color=color, width=2), row=2, col=1)
    if has_strength:
        fig.add_trace(go.Scatter(x=times, y=np.max(water[:, bins], axis=1), mode="lines",
                                 name="In-band peak", line=dict(color="#655192")), row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text=meta.get("spectrum_unit", "Relative dB"), row=3, col=1)
    fig.update_annotations(font_size=14, xanchor="left", x=0)
    fig.update_layout(height=220*len(titles), showlegend=False, margin=dict(l=15,r=12,t=34,b=38))
    return fig
