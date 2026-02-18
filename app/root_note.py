from __future__ import annotations

import math
from typing import Iterable

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Compute weighted median in O(N log N)."""
    if not values or not weights or len(values) != len(weights):
        return float("nan")
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float("nan")
    v = v[valid]
    w = w[valid]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    csum = np.cumsum(w)
    cutoff = 0.5 * csum[-1]
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def estimate_root_note(frames: Iterable[dict]) -> tuple[str | None, float | None]:
    """
    Estimate track root note by weighted pitch-class histogram and weighted median F*.
    """
    hist = np.zeros(12, dtype=np.float64)
    values: list[float] = []
    weights: list[float] = []

    for frame in frames:
        if not frame.get("valid", False):
            continue

        f0 = float(frame.get("f0", -1.0))
        rms = float(frame.get("rms", 0.0))
        conf = float(frame.get("conf", 0.0))
        if f0 <= 0.0 or not np.isfinite(f0):
            continue
        if not np.isfinite(rms) or not np.isfinite(conf):
            continue

        w = (rms**1.5) * (conf**2.0)
        if not np.isfinite(w) or w <= 0.0:
            continue

        midi_float = 69.0 + 12.0 * math.log2(f0 / 440.0)
        midi = int(round(midi_float))
        pitch_class = (midi % 12 + 12) % 12

        hist[pitch_class] += w
        values.append(f0)
        weights.append(w)

    if not values:
        return None, None

    root_pitch_class = int(np.argmax(hist))
    root_note = NOTE_NAMES[root_pitch_class]
    f_star = weighted_median(values, weights)
    if not np.isfinite(f_star) or f_star <= 0.0:
        return root_note, None
    return root_note, float(f_star)


def nearest_note_and_deviation(f0_hz: float) -> tuple[str | None, float | None, int | None]:
    """Compute nearest tempered note and deviations in Hz/cents."""
    if f0_hz <= 0.0 or not np.isfinite(f0_hz):
        return None, None, None

    nearest_midi = int(round(69.0 + 12.0 * math.log2(f0_hz / 440.0)))
    note_name = NOTE_NAMES[(nearest_midi % 12 + 12) % 12]
    octave = math.floor(nearest_midi / 12) - 1
    nearest_note = f"{note_name}{octave}"

    nearest_freq = 440.0 * (2.0 ** ((nearest_midi - 69) / 12.0))
    deviation_hz = float(f0_hz - nearest_freq)
    deviation_cents = int(round(1200.0 * math.log2(f0_hz / nearest_freq)))
    return nearest_note, deviation_hz, deviation_cents
