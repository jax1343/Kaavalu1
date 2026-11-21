"""Offline band and time detection using two-sided Hann-windowed spectra."""
import numpy as np

from .iq_loader import check_receiver

NFFT = 512
HOP = NFFT // 2


def _runs(mask, gap=0):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > gap + 1) + 1)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _db(power):
    return 10 * np.log10(np.maximum(power, 1e-30))


def _pool(values, axis, limit):
    count = values.shape[axis]
    if count <= limit:
        return values
    groups = np.array_split(np.arange(count), limit)
    return np.stack([np.take(values, group, axis=axis).mean(axis=axis) for group in groups], axis=axis)


def detect_signals(iq_data: np.ndarray, sample_rate_hz: float, center_freq_hz: float,
                   min_snr_db: float = 6.0) -> tuple[list[dict], dict]:
    check_receiver(sample_rate_hz, center_freq_hz)
    x = np.asarray(iq_data)
    if x.ndim != 1 or not np.iscomplexobj(x) or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty one-dimensional complex IQ samples")
    if not np.isfinite(min_snr_db):
        raise ValueError("min_snr_db must be finite")
    fs = float(sample_rate_hz)
    df = fs / NFFT
    freq = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs))
    padded = np.pad(x, (0, max(0, NFFT - len(x))))
    starts = np.arange(0, len(padded) - NFFT + 1, HOP)
    if starts[-1] != len(padded) - NFFT:
        starts = np.append(starts, len(padded) - NFFT)
    window = np.hanning(NFFT)
    scale = fs * np.dot(window, window)
    # Store densities as float32. Bound the temporary complex FFT workspace.
    psd = np.empty((len(starts), NFFT), dtype=np.float32)
    for first in range(0, len(starts), 256):
        block = starts[first:first+256]
        frames = padded[block[:, None] + np.arange(NFFT)] * window
        psd[first:first+len(block)] = np.abs(np.fft.fftshift(np.fft.fft(frames), axes=1)) ** 2 / scale
    densities = psd.mean(axis=0, dtype=np.float64)
    floor = max(float(np.median(densities)), 1e-30)
    display = dict(source_tag=None, sample_rate_hz=fs, center_freq_hz=float(center_freq_hz),
                   frequency_hz=_pool(center_freq_hz + freq, 0, 256).tolist(),
                   time_s=_pool((starts + NFFT/2) / fs, 0, 160).tolist(),
                   waterfall_db=_db(_pool(_pool(psd, 1, 256), 0, 160)).tolist(),
                   spectrum_freq_hz=(center_freq_hz + freq).tolist(), spectrum_db=_db(densities).tolist(),
                   noise_floor_db_per_hz=float(_db(floor)), spectrum_unit="dB re 1 IQ-unit²/Hz")
    signals = []
    if len(x) < 2048 or np.max(densities) <= 1e-30:
        return signals, display
    # Pool four adjacent frames to discover transient occupied bands. Hysteresis
    # over time then limits each candidate to an interval with a strong seed.
    size = min(4, len(psd))
    blocks = [psd[i:i+size].mean(axis=0) for i in range(0, len(psd)-size+1, size)]
    activity = np.maximum.reduce(blocks)
    smoothed = np.convolve(activity, np.ones(3)/3, "same")
    for lo, hi in _runs(smoothed > 3.6 * floor, gap=3):
        if activity[lo:hi].max() <= 10**0.8 * floor:
            continue
        lo, hi = max(0, lo-2), min(NFFT, hi+2)
        band = psd[:, lo:hi]
        ratio = band.mean(axis=1) / floor
        for t0, t1 in _runs(ratio > 2.3, gap=1):
            start = int(starts[t0])
            stop = min(len(x), int(starts[t1-1]) + NFFT)
            if stop-start < 2048 or np.max(band[t0:t1]) <= 10**0.8 * floor:
                continue
            burst_psd = band[t0:t1].mean(axis=0, dtype=np.float64)
            snr = float(_db(max(burst_psd.mean()/floor - 1, 0)))
            if snr < min_snr_db:
                continue
            excess = np.maximum(burst_psd - floor, 0)
            cdf = np.cumsum(excess) / excess.sum()
            left, right = np.searchsorted(cdf, [0.005, 0.995])
            bw = float((right-left+1)*df)
            offset = float(np.average(freq[lo:hi], weights=excess))
            # Rectangular FFT channel filter on a zero-padded interval, then mix
            # to the measured centroid. Length and sampling rate are preserved.
            raw = x[start:stop]
            n = 1 << int(np.ceil(np.log2(2*len(raw))))
            bins = np.fft.fftfreq(n, 1/fs)
            keep = (bins >= freq[lo]-df/2) & (bins < freq[hi-1]+df/2)
            isolated = np.fft.ifft(np.fft.fft(raw, n)*keep)[:len(raw)]
            baseband = (isolated*np.exp(-2j*np.pi*offset*np.arange(len(raw))/fs)).astype(np.complex64)
            points = baseband[np.linspace(0, len(baseband)-1, min(256,len(baseband)), dtype=int)]
            signals.append(dict(signal_id=f"b{len(signals)+1}", source_tag=None, iq_baseband=baseband,
                                sample_rate_hz=fs, tuning_freq_hz=float(center_freq_hz), offset_hz=offset,
                                center_freq_hz=float(center_freq_hz+offset), occupied_bw_hz=bw,
                                snr_db=snr, sample_count=len(raw), start_sample=start, stop_sample=stop,
                                start_s=start/fs, duration_s=len(raw)/fs, peak_power_dbm=None,
                                calibration_status="UNCALIBRATED", constellation_i=points.real.tolist(),
                                constellation_q=points.imag.tolist()))
    return signals, display
