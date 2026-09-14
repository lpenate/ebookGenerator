"""Escritura de WAV y empaquetado con ffmpeg: capítulos AAC (.m4a) y audiolibro .m4b con capítulos y portada."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import BookMeta


class FfmpegMissing(Exception):
    pass


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FfmpegMissing("ffmpeg/ffprobe no encontrados. Instálalos con: brew install ffmpeg")


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} falló:\n{result.stderr[-2000:]}")


def wav_to_m4a(wav: Path, m4a: Path, bitrate: str = "64k", title: str | None = None, track: int | None = None, meta: BookMeta | None = None) -> None:
    require_ffmpeg()
    m4a.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", "-b:a", bitrate, "-ar", "24000", "-ac", "1"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if track:
        cmd += ["-metadata", f"track={track}"]
    if meta:
        cmd += ["-metadata", f"album={meta.title}"]
        if meta.author:
            cmd += ["-metadata", f"artist={meta.author}", "-metadata", f"album_artist={meta.author}"]
    cmd += ["-movflags", "+faststart", str(m4a)]
    _run(cmd)


def duration_seconds(path: Path) -> float:
    require_ffmpeg()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


@dataclass
class AudiobookChapter:
    title: str
    file: Path


def build_m4b(
    chapters: list[AudiobookChapter],
    output: Path,
    meta: BookMeta,
    cover: Path | None = None,
) -> list[tuple[str, float, float]]:
    """Une los .m4a en un .m4b con marcadores de capítulo, metadatos y portada.

    Devuelve la lista (título, inicio_s, fin_s) de capítulos.
    """
    require_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    timeline: list[tuple[str, float, float]] = []
    cursor = 0.0
    for ch in chapters:
        dur = duration_seconds(ch.file)
        timeline.append((ch.title, cursor, cursor + dur))
        cursor += dur

    with tempfile.TemporaryDirectory(prefix="ebook-m4b-") as tmp:
        tmp_path = Path(tmp)
        concat_list = tmp_path / "list.txt"
        concat_list.write_text(
            "\n".join(f"file '{str(ch.file.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for ch in chapters),
            encoding="utf-8",
        )

        metadata = tmp_path / "metadata.txt"
        lines = [";FFMETADATA1", f"title={_escape(meta.title)}", f"album={_escape(meta.title)}", "genre=Audiobook", "media_type=2"]
        if meta.author:
            lines += [f"artist={_escape(meta.author)}", f"album_artist={_escape(meta.author)}", f"composer={_escape(meta.author)}"]
        if meta.language:
            lines.append(f"language={_escape(meta.language)}")
        for title, start, end in timeline:
            lines += ["", "[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}", f"END={int(end * 1000)}", f"title={_escape(title)}"]
        metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cover_jpg = None
        if cover and cover.exists():
            cover_jpg = tmp_path / "cover.jpg"
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(cover), "-vf", "scale='min(1400,iw)':-2", "-q:v", "3", str(cover_jpg)])

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(metadata)]
        if cover_jpg:
            cmd += ["-i", str(cover_jpg)]
        cmd += ["-map_metadata", "1", "-map", "0:a", "-c:a", "copy"]
        if cover_jpg:
            cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
        cmd += ["-f", "mp4", "-brand", "M4B ", "-movflags", "+faststart", str(output)]
        _run(cmd)

    return timeline


def _escape(value: str) -> str:
    """Escapa los caracteres especiales del formato FFMETADATA."""
    backslash = chr(92)
    for ch in (backslash, "=", ";", "#"):
        value = value.replace(ch, backslash + ch)
    return value.replace("\n", " ")
