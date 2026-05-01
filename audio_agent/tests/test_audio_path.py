"""Tests for audio input path resolution."""

from __future__ import annotations

import wave
import sys
import types

from audio_agent.utils.audio_path import is_ark_offset_path, resolve_audio_input_path


def _write_wav(path: str) -> bytes:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16)

    with open(path, "rb") as wav_file:
        return wav_file.read()


def test_materializes_riff_wav_from_ark_offset(tmp_path):
    source_wav = tmp_path / "source.wav"
    wav_bytes = _write_wav(str(source_wav))

    ark_path = tmp_path / "data_wav.ark"
    prefix = b"not-a-wav-prefix"
    ark_path.write_bytes(prefix + wav_bytes + b"suffix")

    resolved = resolve_audio_input_path(f"{ark_path}:{len(prefix)}")

    with open(resolved, "rb") as resolved_file:
        assert resolved_file.read() == wav_bytes


def test_detects_ark_offset_path():
    assert is_ark_offset_path("data_wav.ark:16742152")
    assert not is_ark_offset_path("normal.wav")


def test_materializes_kaldiio_waveform_from_ark_offset(tmp_path, monkeypatch):
    ark_path = tmp_path / "data_wav.ark"
    ark_path.write_bytes(b"not-riff")

    fake_kaldiio = types.SimpleNamespace(
        load_mat=lambda spec: (8000, [0.0, 0.5, -0.5])
    )
    monkeypatch.setitem(sys.modules, "kaldiio", fake_kaldiio)

    resolved = resolve_audio_input_path(f"{ark_path}:0")

    with wave.open(resolved, "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 3
