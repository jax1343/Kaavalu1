"""Shared RF measurement fields for the three-laptop handover (Python 3.11).

The shared record uses kHz. Laptop 2's native RF record uses Hz and goes
through to_shared_record at the classifier handoff. A missing calibration is encoded as
None in Python, null in JSON, and NULL in a nullable numeric storage column.
Other detector fields pass through validation unchanged.
"""

from collections.abc import Callable, Mapping
from math import isfinite
from numbers import Real
from typing import Literal, TypedDict, TypeVar, cast


class DetectionMeasurements(TypedDict):
    occupied_bw_khz: float
    peak_power_dbm: float | None


def _finite_float(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def validate_measurements(record: Mapping[str, object]) -> DetectionMeasurements:
    """Validate a copy of the outgoing record. No unit conversion is applied."""
    for field in ("occupied_bw_khz", "peak_power_dbm"):
        if field not in record:
            raise ValueError(f"required field {field} is missing")
    bandwidth = _finite_float("occupied_bw_khz", record["occupied_bw_khz"])
    if bandwidth < 0:
        raise ValueError("occupied_bw_khz must be nonnegative")
    power = record["peak_power_dbm"]
    validated = dict(record)
    validated["occupied_bw_khz"] = bandwidth
    validated["peak_power_dbm"] = (
        None if power is None else _finite_float("peak_power_dbm", power)
    )
    return cast(DetectionMeasurements, validated)


def to_shared_record(record: Mapping[str, object]) -> DetectionMeasurements:
    """Accept a native or shared record and emit only the shared bandwidth key."""
    if "occupied_bw_hz" not in record:
        return validate_measurements(record)
    bandwidth_khz = _finite_float("occupied_bw_hz", record["occupied_bw_hz"]) / 1000
    shared = dict(record)
    shared.pop("occupied_bw_hz")
    if "occupied_bw_khz" in shared:
        validated = validate_measurements(shared)
        if validated["occupied_bw_khz"] != bandwidth_khz:
            raise ValueError("occupied_bw_hz and occupied_bw_khz disagree")
        return validated
    shared["occupied_bw_khz"] = bandwidth_khz
    return validate_measurements(shared)


def power_display_text(power_dbm: float | None) -> str:
    if power_dbm is None:
        return "Power unavailable."
    return f"{power_dbm} dBm"


_Result = TypeVar("_Result")


def evaluate_required_power(
    power_dbm: float | None, check: Callable[[float], _Result]
) -> _Result | Literal["UNKNOWN"]:
    """Return UNKNOWN for human review when this rule requires missing power."""
    if power_dbm is None:
        return "UNKNOWN"
    return check(power_dbm)
