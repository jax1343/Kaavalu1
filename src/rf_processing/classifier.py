"""Trained, CPU-only synthetic reference model. No network or deserialization."""
from functools import lru_cache
import numpy as np

from .dataset_generator import CLASSES, simulate_recording
from .feature_extractor import FEATURE_NAMES, extract_features
from .iq_loader import analyze_recording

MODEL_ID = "kaavalu-synthetic-knn-lof-v1"
NOVELTY_CUTOFF = -0.08


def _number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool,np.bool_)) and np.isfinite(v)


def _vector(features):
    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
        raise ValueError("Missing or unexpected waveform features")
    values = [features[k] for k in FEATURE_NAMES]
    if not all(_number(v) for v in values):
        raise ValueError("Nonfinite or nonnumeric waveform feature")
    return np.asarray(values, dtype=float)


def train_reference_model():
    """One training slice per independent single-emitter session, seeds 10000..10287."""
    vectors, labels = [], []
    for c, family in enumerate(CLASSES):
        for j in range(48):
            rec = simulate_recording(family, 10000 + c*48 + j, added_noise_db=(j%6)*1.9)
            bursts, _ = analyze_recording(rec)
            if len(bursts) != 1 or bursts[0]["snr_db"] < 8:
                continue
            row, = extract_features(bursts, rec["sample_rate_hz"])
            vectors.append(_vector(row["features"]))
            labels.append(family)
    if set(labels) != set(CLASSES):
        raise RuntimeError("Insufficient classes detected during reference training")
    matrix = np.stack(vectors)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-12)
    train = (matrix-mean)/scale
    # Local Outlier Factor: reachability distances, local densities and an
    # empirical lower-tail offset fitted only on the training sessions.
    dist = np.linalg.norm(train[:,None]-train[None,:], axis=2)
    np.fill_diagonal(dist, np.inf)
    k = 18
    idx = np.argsort(dist, axis=1)[:,:k]
    kdist = np.take_along_axis(dist, idx[:,-1:], axis=1).ravel()
    reach = np.maximum(np.take_along_axis(dist, idx, axis=1), kdist[idx])
    lrd = 1/np.maximum(reach.mean(axis=1), 1e-12)
    offset = np.quantile(-lrd[idx].mean(axis=1)/lrd, .03)
    return dict(model_id=MODEL_ID, feature_names=FEATURE_NAMES, train=train, labels=np.array(labels),
                mean=mean, scale=scale, kdist=kdist, lrd=lrd, offset=float(offset))


@lru_cache(maxsize=1)
def _default_bundle():
    return train_reference_model()


def _evaluate(vector, bundle):
    if not isinstance(bundle, dict) or bundle.get("model_id") != MODEL_ID:
        raise ValueError("Missing or unsupported trained model bundle")
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Incompatible model feature schema")
    train = bundle["train"]
    labels = bundle["labels"]
    if train.ndim != 2 or train.shape[1] != len(FEATURE_NAMES) or len(train) < 18 or len(labels) != len(train):
        raise ValueError("Invalid trained reference matrix")
    distance = np.linalg.norm(train-(vector-bundle["mean"])/bundle["scale"], axis=1)
    near = np.argsort(distance)[:18]
    reach = np.maximum(distance[near], bundle["kdist"][near])
    novelty = -bundle["lrd"][near].mean()*reach.mean() - bundle["offset"]
    voters = near[:7]
    weights = 1/np.maximum(distance[voters], 1e-8)**2
    votes = np.array([weights[labels[voters]==name].sum() for name in CLASSES])
    if not votes.sum() > 0 or not np.isfinite([novelty, *votes]).all():
        raise ValueError("Invalid trained model scores")
    best = votes.argmax()
    return CLASSES[best], float(votes[best]/votes.sum()), float(novelty)


def classify_signals(feature_records: list[dict], model_bundle: dict = None,
                     conf_threshold: float = 0.68) -> list[dict]:
    """None lazily trains and caches the real synthetic model. Confidence is weighted vote share."""
    if not _number(conf_threshold) or not 0 <= conf_threshold <= 1:
        raise ValueError("conf_threshold must be between zero and one")
    out = []
    for record in feature_records:
        row = dict(record)
        row.update(predicted_modulation="UNKNOWN", classification_confidence=None, is_unknown=True,
                   lof_normality_score=None, clf_reason="", rule_family="UNKNOWN",
                   candidate_modulation=None, model_id=None)
        row.setdefault("features", {})
        snr, count = row.get("snr_db"), row.get("sample_count")
        if not _number(snr):
            reason = "Missing or invalid measured SNR"
        elif snr < 8.0:
            reason = f"Low SNR ({snr:.3f} dB < 8.0 dB)"
        elif not isinstance(count, (int, np.integer)) or count < 2048:
            reason = "Missing sample count or fewer than 2048 samples"
        elif row.get("feature_error"):
            reason = row["feature_error"]
        else:
            try:
                vec = _vector(row["features"])
                bundle = _default_bundle() if model_bundle is None else model_bundle
                label, confidence, novelty = _evaluate(vec, bundle)
                row.update(candidate_modulation=label, classification_confidence=confidence,
                           lof_normality_score=novelty, model_id=MODEL_ID)
                if novelty < NOVELTY_CUTOFF:
                    reason = f"Novel waveform (LOF {novelty:.3f} < {NOVELTY_CUTOFF})"
                elif confidence < conf_threshold:
                    reason = f"Ambiguous vote ({confidence:.3f} < {conf_threshold:.3f})"
                else:
                    row.update(predicted_modulation=label, rule_family=label, is_unknown=False)
                    reason = "Accepted by SNR, familiarity and confidence checks"
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, RuntimeError) as exc:
                reason = f"Insufficient evidence: {exc}"
        row["clf_reason"] = reason
        out.append(row)
    return out
