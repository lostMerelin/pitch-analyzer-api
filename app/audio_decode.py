from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioDecodeError(Exception):
    """Domain error for decoding failures with an HTTP-friendly status code."""

    def __init__(self, message: str, status_code: int = 415) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def decode_to_mono_float32(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """
    Decode uploaded audio to mono float32 PCM at 44.1 kHz using ffmpeg.

    Args:
        file_bytes: Raw uploaded file bytes.
        filename: Original uploaded filename.

    Returns:
        (samples, sample_rate) where samples is float32 mono PCM.

    Raises:
        AudioDecodeError: If ffmpeg is missing or decoding fails.
    """
    if not file_bytes:
        raise AudioDecodeError("Uploaded file is empty.", status_code=415)

    ext = Path(filename or "").suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="pitch-analyzer-") as tmp_dir:
        input_path = Path(tmp_dir) / f"input{ext}"
        output_path = Path(tmp_dir) / "decoded.wav"
        input_path.write_bytes(file_bytes)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-f",
            "wav",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise AudioDecodeError("ffmpeg is not installed on the server.", status_code=500) from exc
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.strip().splitlines()
            reason = details[-1] if details else "unknown ffmpeg error"
            raise AudioDecodeError(f"Could not decode audio file: {reason}", status_code=415) from exc

        try:
            samples, sr = sf.read(output_path, dtype="float32", always_2d=False)
        except Exception as exc:  # pragma: no cover - depends on decoder backends.
            raise AudioDecodeError("Decoded WAV could not be read.", status_code=500) from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    if samples.size == 0:
        raise AudioDecodeError("Decoded audio is empty.", status_code=415)

    return samples, int(sr)
