from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio import AudiobookChapter, build_m4b, wav_to_m4a, write_wav
from .epub import extract_epub
from .models import Book, Chapter
from .tts_xtts import SAMPLE_RATE, XttsEngine

Logger = Callable[[str], None]


def slugify(text: str, max_length: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:max_length].rstrip("-") or "sin-titulo"


def chapter_basename(chapter: Chapter, total: int) -> str:
    width = max(2, len(str(total)))
    return f"{chapter.index:0{width}d}-{slugify(chapter.title)}"


@dataclass
class PipelineOptions:
    epub_path: Path
    out_dir: Path
    min_words: int = 100
    toc_depth: int = 1
    only: set[int] | None = None
    force: bool = False
    audiobook: bool = True
    bitrate: str = "64k"
    keep_wav: bool = False


def prepare(options: PipelineOptions, log: Logger) -> tuple[Book, Path]:
    """Extrae el libro, escribe textos y portada. Devuelve el libro y su directorio de salida."""
    book = extract_epub(options.epub_path, min_words=options.min_words, toc_depth=options.toc_depth)
    out = options.out_dir / slugify(book.meta.title)
    (out / "text").mkdir(parents=True, exist_ok=True)

    author = f" — {book.meta.author}" if book.meta.author else ""
    log(f"[bold]{book.meta.title}[/bold]{author}")
    log(f"Capítulos: {len(book.chapters)} (descartados por cortos: {len(book.skipped)})")

    for chapter in book.chapters:
        path = out / "text" / f"{chapter_basename(chapter, len(book.chapters))}.txt"
        path.write_text(f"{chapter.title}\n\n{chapter.text}\n", encoding="utf-8")

    if book.cover:
        cover_path = out / f"cover.{book.cover.extension}"
        cover_path.write_bytes(book.cover.data)
        log(f"Portada: {cover_path.name} ({len(book.cover.data) // 1024} KB)")
    else:
        log("[yellow]Sin portada en el EPUB[/yellow]")
    return book, out


def cover_path_for(book: Book, out: Path) -> Path | None:
    return (out / f"cover.{book.cover.extension}") if book.cover else None


ChapterHook = Callable[[Chapter, str], None]  # estados: "start" | "done" | "skipped"


def synthesize(
    book: Book,
    out: Path,
    engine: XttsEngine,
    options: PipelineOptions,
    log: Logger,
    progress: Callable[[int, int], None] | None = None,
    on_chapter: ChapterHook | None = None,
) -> dict[int, Path]:
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    targets = [c for c in book.chapters if not options.only or c.index in options.only]

    log(f"Cargando XTTS v2 en [bold]{engine.device}[/bold]…")
    t0 = time.time()
    engine.load()
    log(f"Modelo listo en {time.time() - t0:.1f}s en [bold]{engine.device_label}[/bold]. Hablante: {engine.speaker or engine.speaker_wav}, idioma: {engine.language}")

    for chapter in targets:
        base = chapter_basename(chapter, len(book.chapters))
        m4a = audio_dir / f"{base}.m4a"
        wav = audio_dir / f"{base}.wav"
        if m4a.exists() and m4a.stat().st_size > 0 and not options.force:
            log(f"  [{chapter.index}] ya existe, se omite: {m4a.name}")
            results[chapter.index] = m4a
            if on_chapter:
                on_chapter(chapter, "skipped")
            continue

        log(f"  [{chapter.index}] {chapter.title} — {chapter.words} palabras")
        if on_chapter:
            on_chapter(chapter, "start")
        started = time.time()
        audio = engine.synthesize_text(f"{chapter.title}.\n\n{chapter.text}", on_progress=progress)
        write_wav(wav, audio, SAMPLE_RATE)
        wav_to_m4a(wav, m4a, bitrate=options.bitrate, title=chapter.title, track=chapter.index, meta=book.meta)
        if not options.keep_wav:
            wav.unlink(missing_ok=True)
        elapsed = time.time() - started
        duration = audio.size / SAMPLE_RATE
        log(f"  [{chapter.index}] ✔ {m4a.name} — {duration / 60:.1f} min de audio en {elapsed / 60:.1f} min ({duration / max(elapsed, 0.01):.2f}x tiempo real)")
        results[chapter.index] = m4a
        if on_chapter:
            on_chapter(chapter, "done")

    # Recoge también audios previos de capítulos no seleccionados en esta pasada.
    for chapter in book.chapters:
        candidate = audio_dir / f"{chapter_basename(chapter, len(book.chapters))}.m4a"
        if chapter.index not in results and candidate.exists():
            results[chapter.index] = candidate
    return results


def pack(book: Book, out: Path, audio_files: dict[int, Path], log: Logger) -> Path | None:
    ready = [c for c in book.chapters if c.index in audio_files]
    missing = [c.index for c in book.chapters if c.index not in audio_files]
    if not ready:
        log("[yellow]No hay audios: no se genera el audiolibro[/yellow]")
        return None
    if missing:
        log(f"[yellow]Faltan audios de los capítulos {missing}; el M4B se genera con los {len(ready)} disponibles[/yellow]")
    output = out / f"{slugify(book.meta.title)}.m4b"
    timeline = build_m4b(
        [AudiobookChapter(title=c.title, file=audio_files[c.index]) for c in ready],
        output,
        book.meta,
        cover=cover_path_for(book, out),
    )
    total = timeline[-1][2] if timeline else 0
    log(f"Audiolibro: {output} ({total / 3600:.2f} h, {len(timeline)} capítulos)")
    return output


def write_manifest(book: Book, out: Path, options: PipelineOptions, engine: XttsEngine | None, audio_files: dict[int, Path], audiobook: Path | None) -> Path:
    manifest = {
        "meta": {"title": book.meta.title, "author": book.meta.author, "language": book.meta.language},
        "source": str(options.epub_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "xtts_v2",
        "device": engine.device_label if engine else None,
        "speaker": (engine.speaker or engine.speaker_wav) if engine else None,
        "cover": cover_path_for(book, out).name if book.cover else None,
        "audiobook": audiobook.name if audiobook else None,
        "chapters": [
            {
                "index": c.index,
                "title": c.title,
                "words": c.words,
                "chars": len(c.text),
                "text_file": f"text/{chapter_basename(c, len(book.chapters))}.txt",
                "audio_file": f"audio/{audio_files[c.index].name}" if c.index in audio_files else None,
            }
            for c in book.chapters
        ],
        "skipped": [{"href": h, "title": t, "words": w} for h, t, w in book.skipped],
    }
    path = out / "book.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
