"""Utilities for resolving audio input paths."""

from __future__ import annotations

import re
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any


_ARK_OFFSET_RE = re.compile(r"^(?P<ark_path>.+\.ark):(?P<offset>\d+)$")


def is_ark_offset_path(path: str) -> bool:
    """Return whether path looks like a Kaldi-style ark byte-offset reference."""
    return _ARK_OFFSET_RE.match(path.strip()) is not None


def resolve_audio_input_path(path: str) -> str:
    """Resolve an audio input path, materializing ark byte-offset WAV entries if needed.

    Supports normal file paths and Kaldi-style references such as
    ``data_wav.ark:16742152``. Kaldi ark entries are read with ``kaldiio`` when
    possible, then written to a temporary WAV file for downstream audio models.
    Offsets that point directly at RIFF/WAV bytes are also supported.
    """
    stripped = path.strip()
    match = _ARK_OFFSET_RE.match(stripped)
    if match is None:
        return str(Path(stripped).resolve())

    ark_path = Path(match.group("ark_path")).resolve()
    offset = int(match.group("offset"))
    return _materialize_wav_from_ark(ark_path, offset)


def resolve_audio_input_paths(paths: list[str]) -> list[str]:
    """Resolve a list of audio input paths."""
    return [resolve_audio_input_path(path) for path in paths]


def _materialize_wav_from_ark(ark_path: Path, offset: int) -> str:
    if not ark_path.exists():
        raise FileNotFoundError(f"ARK file not found: {ark_path}")
    if offset < 0:
        raise ValueError(f"ARK offset must be non-negative: {offset}")

    try:
        return _materialize_kaldi_wav_with_kaldiio(ark_path, offset)
    except Exception as kaldi_error:
        try:
            return _materialize_embedded_riff_wav(ark_path, offset)
        except Exception as riff_error:
            raise ValueError(
                f"Failed to read Kaldi ark audio at {ark_path}:{offset}. "
                "The entry is neither a kaldiio-readable waveform nor an embedded RIFF/WAV "
                f"payload. kaldiio error: {kaldi_error}; RIFF fallback error: {riff_error}"
            ) from riff_error


def _materialize_embedded_riff_wav(ark_path: Path, offset: int) -> str:
    with ark_path.open("rb") as source:
        source.seek(offset)
        header = source.read(12)
        if len(header) < 12:
            raise ValueError(f"ARK offset is beyond readable data: {ark_path}:{offset}")
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(
                "Offset does not point directly to embedded RIFF/WAV data: "
                f"{ark_path}:{offset}"
            )

        riff_size = struct.unpack("<I", header[4:8])[0]
        total_size = riff_size + 8
        source.seek(offset)
        wav_bytes = source.read(total_size)
        if len(wav_bytes) != total_size:
            raise ValueError(
                f"Incomplete WAV payload at {ark_path}:{offset}; "
                f"expected {total_size} bytes, got {len(wav_bytes)}"
            )

    output = tempfile.NamedTemporaryFile(
        prefix=f"{ark_path.stem}_{offset}_",
        suffix=".wav",
        delete=False,
    )
    with output:
        output.write(wav_bytes)

    return output.name


def _materialize_kaldi_wav_with_kaldiio(ark_path: Path, offset: int) -> str:
    try:
        import kaldiio
    except ImportError as exc:
        raise RuntimeError("kaldiio is required to read non-RIFF Kaldi ark audio") from exc

    obj = kaldiio.load_mat(f"{ark_path}:{offset}")
    sample_rate, waveform = _normalize_kaldiio_wave_object(obj)
    return _write_waveform_to_temp_wav(waveform, sample_rate, ark_path.stem, offset)


def _normalize_kaldiio_wave_object(obj: Any) -> tuple[int, Any]:
    """Normalize kaldiio waveform output to (sample_rate, waveform)."""
    if isinstance(obj, tuple) and len(obj) == 2:
        first, second = obj
        if isinstance(first, (int, float)):
            return int(first), second
        if isinstance(second, (int, float)):
            return int(second), first

    raise TypeError(
        "kaldiio did not return a waveform tuple of (sample_rate, waveform) "
        f"or (waveform, sample_rate); got {type(obj).__name__}"
    )


def _write_waveform_to_temp_wav(
    waveform: Any,
    sample_rate: int,
    ark_stem: str,
    offset: int,
) -> str:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to materialize Kaldi ark audio") from exc

    array = np.asarray(waveform)
    if array.size == 0:
        raise ValueError("Kaldi ark waveform is empty")

    if array.ndim == 1:
        channels = 1
    elif array.ndim == 2:
        # Kaldi wave data is commonly shaped as (channels, samples). Convert to
        # wave's interleaved (samples, channels) shape when needed.
        if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
            array = array.T
        channels = array.shape[1]
    else:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, -1.0, 1.0)
        pcm = (array * 32767.0).astype("<i2")
    elif array.dtype == np.int16:
        pcm = array.astype("<i2", copy=False)
    else:
        info = np.iinfo(array.dtype) if np.issubdtype(array.dtype, np.integer) else None
        if info is not None:
            scale = max(abs(info.min), abs(info.max))
            pcm = (array.astype("float32") / scale * 32767.0).astype("<i2")
        else:
            raise ValueError(f"Unsupported waveform dtype: {array.dtype}")

    output = tempfile.NamedTemporaryFile(
        prefix=f"{ark_stem}_{offset}_",
        suffix=".wav",
        delete=False,
    )
    output_path = output.name
    output.close()

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return output_path
