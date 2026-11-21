"""Waveform-only features of the measured, isolated baseband slice."""
import numpy as np

FEATURE_NAMES = ("spectral_flatness", "crest_factor_db", "amp_std_norm", "amp_kurtosis",
                 "phase_std_rad", "inst_freq_var_norm", "cumulant_c40_mag",
                 "cumulant_c42_mag", "duty_cycle")


def _shape(x):
    z = x.astype(np.complex128)
    rms = np.sqrt(np.mean(np.abs(z)**2))
    if rms <= 1e-15:
        raise ValueError("No usable sample energy")
    z /= rms
    amp = np.abs(z)
    segments = np.lib.stride_tricks.sliding_window_view(z, 256)[::128]
    psd = np.mean(np.abs(np.fft.fft(segments * np.hanning(256), axis=1))**2, axis=0)
    psd = np.maximum(psd / psd.mean(), 1e-12)
    phase_steps = np.angle(z[1:] * z[:-1].conj())
    spread = float(amp.std())
    kurt = np.mean((amp-amp.mean())**4)/spread**4 if spread > 1e-10 else 0.
    # Local cumulants limit rotation across a block from residual carrier offset.
    blocks = z[:(len(z)//128)*128].reshape(-1, 128)
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    m20 = (blocks**2).mean(axis=1)
    m21 = (abs(blocks)**2).mean(axis=1)
    c40 = np.abs((blocks**4).mean(axis=1) - 3*m20**2).mean()
    c42 = np.abs((abs(blocks)**4).mean(axis=1) - abs(m20)**2 - 2*m21**2).mean()
    # Circular phase dispersion, radians, capped by a numeric resultant floor.
    strong = amp > 0.1
    resultant = np.clip(np.abs((z[strong]/amp[strong]).mean()), 1e-12, 1.)
    values = [np.exp(np.log(psd).mean()), 20*np.log10(amp.max()), spread/amp.mean(),
              kurt, np.sqrt(-2*np.log(resultant)), (phase_steps/(2*np.pi)).var(),
              c40, c42, np.mean(amp > 0.25)]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite waveform feature")
    return dict(zip(FEATURE_NAMES, map(float, values)))


def extract_features(signals: list[dict], sample_rate_hz: float) -> list[dict]:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    records = []
    for signal in signals:
        row = {key: value for key, value in signal.items() if key != "iq_baseband"}
        x = np.asarray(signal.get("iq_baseband"))
        row["features"] = {}
        if x.ndim != 1 or not np.iscomplexobj(x) or not np.isfinite(x).all():
            row["feature_error"] = "Missing or invalid complex IQ slice"
        elif signal.get("sample_count", len(x)) != len(x):
            row["feature_error"] = "Sample count disagrees with IQ slice"
        elif signal.get("sample_rate_hz", sample_rate_hz) != sample_rate_hz:
            row["feature_error"] = "Sample rate disagrees with IQ slice"
        else:
            row["sample_count"] = len(x)
            if len(x) < 2048:
                row["feature_error"] = "Fewer than 2048 samples"
            else:
                try:
                    row["features"] = _shape(x)
                except ValueError as exc:
                    row["feature_error"] = str(exc)
        records.append(row)
    return records
