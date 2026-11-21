"""Independent seeded SIMULATED captures. Truth remains with experiment callers."""
import numpy as np

from .iq_loader import analyze_recording

CLASSES = ("GFSK", "FM-NB", "BPSK", "QPSK", "AM-DSB", "WB-FM")
FS = 192_000.
TUNING = 150_000_000.
SCENARIOS = {
    "familiar_clear": ("QPSK", 81001),
    "familiar_weak": ("QPSK", 81001),
    "unfamiliar_clear": ("16-QAM", 91001),
    "rescue_corridor": ("FM-NB", 82001),
}


def _waveform(family, rng, length, fs):
    sps = int(rng.integers(9, 16))
    count = int(np.ceil(length/sps)) + 16
    if family in {"BPSK", "QPSK", "16-QAM"}:
        if family == "16-QAM":
            alphabet = [-3, -1, 1, 3]
            symbols = rng.choice(alphabet, count) + 1j*rng.choice(alphabet, count)
        else:
            order = 2 if family == "BPSK" else 4
            symbols = np.exp(2j*np.pi*rng.integers(order, size=count)/order)
        stream = np.zeros(count*sps, complex)
        stream[::sps] = symbols
        u = np.arange(-4*sps, 4*sps+1) / sps
        tap = np.sinc(u) * np.kaiser(len(u), 6.)
        wave = np.convolve(stream, tap, "same")[4*sps:4*sps+length]
    else:
        t = np.arange(length) / fs
        tones = rng.uniform(450, 2350, size=5)
        audio = np.sin(2*np.pi*t[:,None]*tones + rng.uniform(0, 2*np.pi, 5)).sum(axis=1)
        audio /= np.max(np.abs(audio))
        if family == "AM-DSB":
            wave = 1 + rng.uniform(.55,.9)*audio
        elif family in {"FM-NB", "WB-FM", "GFSK"}:
            if family == "GFSK":
                data = np.repeat(rng.choice([-1.,1.], count), sps)
                taps = np.exp(-.5*(np.arange(-sps,sps+1)/(.33*sps))**2)
                deviation = np.convolve(data, taps/taps.sum(), "same")[:length]*fs/(4*sps)
            else:
                depth = rng.uniform(2300,4600) if family == "FM-NB" else rng.uniform(14000,23000)
                deviation = depth*audio
            wave = np.exp(2j*np.pi*np.cumsum(deviation)/fs)
        else:
            raise ValueError(f"Unknown simulation waveform: {family}")
    return wave / np.sqrt(np.mean(np.abs(wave)**2))


def simulate_recording(family, seed, added_noise_db=0., *, tuning_hz=TUNING, offset_hz=None):
    """Return IQ and acquisition metadata only. Positive added_noise_db increases noise power."""
    if not np.isfinite(added_noise_db) or not -80 <= added_noise_db <= 90:
        raise ValueError("added_noise_db must be finite and between -80 and 90")
    rng = np.random.default_rng(seed)
    length = 12288
    start = int(rng.integers(400, 1200))
    duration = int(rng.integers(8400, 10200))
    offset = rng.uniform(-35000, 35000) if offset_hz is None else float(offset_hz)
    wave = _waveform(family, rng, duration, FS)
    carrier = np.exp(1j*(2*np.pi*offset*np.arange(duration)/FS + rng.uniform(0, 2*np.pi)))
    noise = (rng.normal(size=length) + 1j*rng.normal(size=length))/np.sqrt(2)
    x = noise*.22*10**(added_noise_db/20)
    x[start:start+duration] += wave*carrier
    return dict(iq_data=x.astype(np.complex64), sample_rate_hz=FS,
                center_freq_hz=float(tuning_hz), source_tag="SIMULATED")


def save_simulated_iq(file_path, recording):
    if recording["source_tag"] != "SIMULATED":
        raise ValueError("Expected SIMULATED recording")
    with open(file_path, "wb") as stream:
        np.savez(stream, **{k: recording[k] for k in ("iq_data","source_tag","sample_rate_hz","center_freq_hz")})


def generate_simulated_disasternet_capture(scenario_name: str, added_noise_db: float = 0.0) -> tuple[list[dict], dict]:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario. Choose from {', '.join(SCENARIOS)}")
    family, seed = SCENARIOS[scenario_name]
    if scenario_name == "familiar_weak":
        # Find measurable weak evidence on the same QPSK waveform as the clear run.
        for extra in np.arange(0., 31., .5):
            capture = simulate_recording(family, seed, added_noise_db+extra, offset_hz=18000.)
            rows, display = analyze_recording(capture)
            if len(rows) == 1 and rows[0]["snr_db"] < 8:
                display["added_noise_db"] = float(added_noise_db + extra)
                return rows, display
            if not rows:
                display["added_noise_db"] = float(added_noise_db + extra)
                return rows, display
        raise ValueError("Noise sweep found neither a weak slice nor a missed detection")
    recording = simulate_recording(family, seed, added_noise_db, offset_hz=18000.)
    return analyze_recording(recording)
