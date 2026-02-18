from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PitchOptions:
    sr: int = 44100
    window: int = 2048
    hop: int = 512
    min_f0: float = 50.0
    max_f0: float = 1200.0
    rms_gate: float = 0.015
    conf_gate: float = 0.80
    median_window: int = 5
    ema_alpha: float = 0.30
    jump_gate_cents: float = 80.0
    octave_search: bool = True
    octave_keep_limit_cents: float = 700.0
    yin_threshold: float = 0.15
    reset_invalid_after: int = 10


def compute_rms(frame: np.ndarray) -> float:
    """Compute RMS for one time-domain frame."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def cents_diff(f1: float, f2: float) -> float:
    """Difference in cents between two frequencies."""
    if f1 <= 0.0 or f2 <= 0.0 or not np.isfinite(f1) or not np.isfinite(f2):
        return np.inf
    return float(1200.0 * np.log2(f1 / f2))


def _yin_difference(frame: np.ndarray, max_tau: int) -> np.ndarray:
    """
    Compute YIN difference function d(tau) in O(N log N) via autocorrelation.
    """
    n = frame.size
    fft_size = 1 << ((2 * n - 1).bit_length())
    spectrum = np.fft.rfft(frame, n=fft_size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size).real[:n]

    sq = np.square(frame, dtype=np.float64)
    csum = np.concatenate((np.array([0.0]), np.cumsum(sq)))
    taus = np.arange(1, max_tau + 1)

    len_terms = n - taus
    sum_x = csum[len_terms]
    sum_y = csum[n] - csum[taus]
    d = sum_x + sum_y - 2.0 * acf[taus]

    out = np.zeros(max_tau + 1, dtype=np.float64)
    out[1:] = np.maximum(d, 0.0)
    return out


def yin_pitch(
    frame: np.ndarray,
    sr: int,
    min_f0: float = 50.0,
    max_f0: float = 1200.0,
    threshold: float = 0.15,
) -> tuple[float, float]:
    """
    YIN pitch detection.

    Returns:
        (f0_hz, confidence) where confidence in [0,1], or (-1,0) if no pitch.
    """
    if frame.size < 32:
        return -1.0, 0.0

    min_tau = int(sr / max_f0)
    max_tau = int(sr / min_f0)
    max_tau = min(max_tau, frame.size - 2)
    min_tau = max(min_tau, 2)
    if max_tau <= min_tau:
        return -1.0, 0.0

    d = _yin_difference(frame, max_tau)
    cmnd = np.ones_like(d)
    cumulative = np.cumsum(d[1:])
    taus = np.arange(1, max_tau + 1, dtype=np.float64)
    denom = np.maximum(cumulative, 1e-12)
    cmnd[1:] = (d[1:] * taus) / denom

    search = cmnd[min_tau : max_tau + 1]
    below = np.where(search < threshold)[0]
    if below.size > 0:
        tau = int(min_tau + below[0])
        while tau + 1 <= max_tau and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    else:
        tau = int(np.argmin(search) + min_tau)

    if tau <= 0 or tau >= max_tau:
        return -1.0, 0.0

    prev_tau = tau - 1
    next_tau = tau + 1
    denom_interp = 2.0 * (2.0 * cmnd[tau] - cmnd[prev_tau] - cmnd[next_tau])
    if abs(denom_interp) > 1e-12:
        delta = (cmnd[next_tau] - cmnd[prev_tau]) / denom_interp
        tau_refined = float(np.clip(tau + delta, min_tau, max_tau))
    else:
        tau_refined = float(tau)

    if tau_refined <= 0.0:
        return -1.0, 0.0

    f0 = float(sr / tau_refined)
    if not np.isfinite(f0) or f0 < min_f0 or f0 > max_f0:
        return -1.0, 0.0

    conf = float(np.clip(1.0 - cmnd[tau], 0.0, 1.0))
    return f0, conf


def _choose_octave_candidate(f0: float, prev_smooth: float, opts: PitchOptions) -> float:
    if not opts.octave_search or prev_smooth <= 0.0:
        return f0

    candidates = [
        c for c in (f0, f0 * 0.5, f0 * 2.0) if opts.min_f0 <= c <= opts.max_f0 and np.isfinite(c)
    ]
    if not candidates:
        return f0

    cents = [abs(cents_diff(c, prev_smooth)) for c in candidates]
    best_idx = int(np.argmin(cents))
    if cents[best_idx] > opts.octave_keep_limit_cents:
        return f0
    return float(candidates[best_idx])


def run_pitch_tracking(samples: np.ndarray, sr: int = 44100, opts: PitchOptions | None = None) -> list[dict]:
    """
    Run frame-based pitch tracking with voiced gate, anti-octave, median+EMA and jump gate.
    """
    if opts is None:
        opts = PitchOptions(sr=sr)
    else:
        opts = PitchOptions(**{**opts.__dict__, "sr": sr})

    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size < opts.window:
        return []

    frames: list[dict] = []
    valid_history: deque[float] = deque(maxlen=opts.median_window)
    prev_smooth = -1.0
    invalid_count = 0

    for start in range(0, x.size - opts.window + 1, opts.hop):
        frame = x[start : start + opts.window]
        t = float(start / sr)
        rms = compute_rms(frame)
        f0_raw, conf = yin_pitch(
            frame,
            sr=sr,
            min_f0=opts.min_f0,
            max_f0=opts.max_f0,
            threshold=opts.yin_threshold,
        )

        valid = (
            np.isfinite(f0_raw)
            and f0_raw > 0.0
            and opts.min_f0 <= f0_raw <= opts.max_f0
            and rms >= opts.rms_gate
            and conf >= opts.conf_gate
        )

        out_f0 = -1.0
        jump = None

        if valid:
            corrected = _choose_octave_candidate(f0_raw, prev_smooth, opts)
            proposed = list(valid_history)
            proposed.append(corrected)
            if len(proposed) > opts.median_window:
                proposed = proposed[-opts.median_window :]

            median_hz = float(np.median(np.asarray(proposed, dtype=np.float64)))
            smooth_hz = median_hz if prev_smooth <= 0.0 else (
                opts.ema_alpha * median_hz + (1.0 - opts.ema_alpha) * prev_smooth
            )

            if prev_smooth > 0.0:
                jump = abs(cents_diff(smooth_hz, prev_smooth))

            if jump is not None and jump > opts.jump_gate_cents:
                valid = False
            else:
                valid_history = deque(proposed, maxlen=opts.median_window)
                prev_smooth = smooth_hz
                out_f0 = smooth_hz
                invalid_count = 0

        if not valid:
            out_f0 = -1.0
            invalid_count += 1
            if invalid_count >= opts.reset_invalid_after:
                valid_history.clear()
                prev_smooth = -1.0
                invalid_count = 0

        frame_result = {
            "t": t,
            "f0": float(out_f0),
            "conf": float(conf),
            "rms": float(rms),
            "valid": bool(valid),
        }
        if jump is not None:
            frame_result["cents_jump"] = float(jump)
        frames.append(frame_result)

    return frames


def downsample_pitch_track(frames: Iterable[dict], step_sec: float = 0.05) -> list[dict]:
    """Return compact pitch track for API response with 10-20 Hz update rate."""
    out: list[dict] = []
    last_t = -1e9
    for frame in frames:
        if not frame.get("valid", False):
            continue
        t = float(frame.get("t", 0.0))
        if t - last_t < step_sec:
            continue
        out.append(
            {
                "t": round(t, 3),
                "f0": round(float(frame.get("f0", -1.0)), 3),
                "conf": round(float(frame.get("conf", 0.0)), 3),
            }
        )
        last_t = t
    return out
