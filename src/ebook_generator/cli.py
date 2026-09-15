from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from . import pipeline
from .epub import extract_epub
from .pipeline import PipelineOptions

app = typer.Typer(help="Convierte un EPUB en audiolibro con XTTS v2: texto y audio por capítulo, portada y M4B.", no_args_is_help=True)
console = Console(stderr=True)
log = console.print


def parse_ranges(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        m = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not m:
            raise typer.BadParameter(f'Rango inválido "{part}". Usa "1-3,7".')
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        result.update(range(min(a, b), max(a, b) + 1))
    return result


@app.command()
def extract(
    epub: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Fichero .epub")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Directorio de salida")] = Path("out"),
    min_words: Annotated[int, typer.Option(help="Palabras mínimas para considerar capítulo")] = 100,
    toc_depth: Annotated[int, typer.Option(help="Nivel de la tabla de contenidos que define un capítulo (1 = primer nivel)")] = 1,
    list_only: Annotated[bool, typer.Option("--list", help="Solo lista capítulos, no escribe nada")] = False,
) -> None:
    """Extrae capítulos a texto y la portada, sin generar audio."""
    if list_only:
        book = extract_epub(epub, min_words=min_words, toc_depth=toc_depth)
        table = Table(title=f"{book.meta.title}" + (f" — {book.meta.author}" if book.meta.author else ""))
        table.add_column("#", justify="right")
        table.add_column("Título")
        table.add_column("Palabras", justify="right")
        for c in book.chapters:
            table.add_row(str(c.index), c.title[:70], f"{c.words:,}")
        total = sum(c.words for c in book.chapters)
        table.add_section()
        table.add_row("", f"Total (~{total / 150:.0f} min de audio)", f"{total:,}")
        console.print(table)
        console.print(f"Portada: {'sí (' + book.cover.media_type + ')' if book.cover else 'no encontrada'}")
        if book.skipped:
            console.print("\n[dim]Descartados por cortos (ajusta con --min-words):[/dim]")
            for href, title, words in book.skipped:
                console.print(f"  [dim]{title or Path(href).name} — {words} palabras[/dim]")
        return
    options = PipelineOptions(epub_path=epub, out_dir=out, min_words=min_words, toc_depth=toc_depth)
    book, out_dir = pipeline.prepare(options, log)
    pipeline.write_manifest(book, out_dir, options, None, {}, None)
    log(f"Texto en {out_dir / 'text'}")


@app.command()
def build(
    epub: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Fichero .epub")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Directorio de salida")] = Path("out"),
    speaker: Annotated[Optional[str], typer.Option("--speaker", "-s", help="Hablante preentrenado de XTTS (ver `voices`)")] = None,
    speaker_wav: Annotated[Optional[Path], typer.Option("--speaker-wav", exists=True, dir_okay=False, help="WAV de 6-30 s para clonar una voz")] = None,
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="Idioma XTTS (por defecto el del EPUB, o 'es')")] = None,
    speed: Annotated[float, typer.Option(help="Velocidad relativa, 1 = normal")] = 1.0,
    device: Annotated[str, typer.Option(help="auto | m1 | mps | cpu | cuda (m1: Apple Silicon, GPT en MPS y vocoder en CPU)")] = "auto",
    chapters: Annotated[Optional[str], typer.Option("--chapters", "-c", help='Capítulos a generar, p. ej. "1-3,7"')] = None,
    min_words: Annotated[int, typer.Option(help="Palabras mínimas para considerar capítulo")] = 100,
    toc_depth: Annotated[int, typer.Option(help="Nivel de la tabla de contenidos que define un capítulo")] = 1,
    bitrate: Annotated[str, typer.Option(help="Bitrate AAC por capítulo")] = "64k",
    force: Annotated[bool, typer.Option("--force", "-f", help="Regenera audios ya existentes")] = False,
    no_audiobook: Annotated[bool, typer.Option("--no-audiobook", help="No generar el M4B final")] = False,
    keep_wav: Annotated[bool, typer.Option(help="Conservar los WAV intermedios")] = False,
) -> None:
    """Extrae capítulos y portada, genera un audio por capítulo con XTTS v2 y empaqueta un M4B."""
    from .tts_xtts import XttsEngine

    options = PipelineOptions(
        epub_path=epub, out_dir=out, min_words=min_words, toc_depth=toc_depth, only=parse_ranges(chapters),
        force=force, audiobook=not no_audiobook, bitrate=bitrate, keep_wav=keep_wav,
    )
    book, out_dir = pipeline.prepare(options, log)
    lang = language or _xtts_language(book.meta.language)
    engine = XttsEngine(speaker=speaker, speaker_wav=str(speaker_wav) if speaker_wav else None, language=lang, speed=speed, device=device)

    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), TimeRemainingColumn(), console=console, transient=True) as progress:
        task = progress.add_task("fragmentos", total=1)

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        audio_files = pipeline.synthesize(book, out_dir, engine, options, log, on_progress)

    audiobook = pipeline.pack(book, out_dir, audio_files, log) if options.audiobook else None
    manifest = pipeline.write_manifest(book, out_dir, options, engine, audio_files, audiobook)
    log(f"Manifiesto: {manifest}")


@app.command()
def pack(
    epub: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Fichero .epub original")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("out"),
    min_words: Annotated[int, typer.Option()] = 100,
    toc_depth: Annotated[int, typer.Option()] = 1,
) -> None:
    """Genera solo el M4B a partir de los audios por capítulo ya existentes."""
    options = PipelineOptions(epub_path=epub, out_dir=out, min_words=min_words, toc_depth=toc_depth)
    book, out_dir = pipeline.prepare(options, log)
    audio_files = {
        c.index: out_dir / "audio" / f"{pipeline.chapter_basename(c, len(book.chapters))}.m4a"
        for c in book.chapters
        if (out_dir / "audio" / f"{pipeline.chapter_basename(c, len(book.chapters))}.m4a").exists()
    }
    audiobook = pipeline.pack(book, out_dir, audio_files, log)
    pipeline.write_manifest(book, out_dir, options, None, audio_files, audiobook)


@app.command()
def voices(
    device: Annotated[str, typer.Option(help="auto | m1 | mps | cpu")] = "cpu",
) -> None:
    """Lista los hablantes preentrenados de XTTS v2."""
    from .tts_xtts import DEFAULT_SPEAKER, XttsEngine

    engine = XttsEngine(device=device)
    for name in engine.speakers():
        marker = "  (por defecto)" if name == DEFAULT_SPEAKER else ""
        typer.echo(f"{name}{marker}")


@app.command()
def sample(
    text: Annotated[str, typer.Argument(help="Texto a sintetizar")] = "Hola. Esta es una prueba de locución con XTTS versión dos en español.",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("out/sample.wav"),
    speaker: Annotated[Optional[str], typer.Option("--speaker", "-s")] = None,
    speaker_wav: Annotated[Optional[Path], typer.Option("--speaker-wav", exists=True)] = None,
    language: Annotated[str, typer.Option("--language", "-l")] = "es",
    speed: Annotated[float, typer.Option()] = 1.0,
    device: Annotated[str, typer.Option(help="auto | m1 | mps | cpu | cuda")] = "auto",
) -> None:
    """Genera un WAV corto para probar un hablante o una voz clonada."""
    import time

    from .audio import write_wav
    from .tts_xtts import SAMPLE_RATE, XttsEngine

    engine = XttsEngine(speaker=speaker, speaker_wav=str(speaker_wav) if speaker_wav else None, language=language, speed=speed, device=device)
    t0 = time.time()
    engine.load()
    log(f"Modelo cargado en {time.time() - t0:.1f}s ({engine.device_label})")
    t0 = time.time()
    audio = engine.synthesize_text(text)
    elapsed = time.time() - t0
    write_wav(out, audio, SAMPLE_RATE)
    log(f"{out} — {audio.size / SAMPLE_RATE:.1f}s de audio en {elapsed:.1f}s")


@app.command()
def kill_port(
    port: Annotated[int, typer.Option("--port", "-p", help="Puerto TCP a limpiar")] = 8000,
    force: Annotated[bool, typer.Option("--force", "-f", help="Usa SIGKILL en lugar de SIGTERM")] = False,
) -> None:
    """Mata los procesos que escuchen el puerto HTTP de la UI (por defecto 8000)."""
    command = shutil.which("lsof")
    if command:
        proc = subprocess.run([command, "-ti", f"tcp:{port}"], text=True, capture_output=True)
        stdout = (proc.stdout or "").strip()
        pids = [int(p) for p in stdout.splitlines() if p.strip().isdigit()]
    elif shutil.which("fuser"):
        proc = subprocess.run([shutil.which("fuser"), "-k", f"{port}/tcp"], text=True, capture_output=True)
        pids = []
        if proc.returncode != 0:
            pids = []
    else:
        typer.echo("No hay lsof ni fuser en PATH; no puedo limpiar el puerto.")
        raise typer.Exit(code=2)

    if not pids:
        typer.echo(f"No hay procesos escuchando el puerto {port}.")
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
    typer.echo(f"Se enviaron {sig.name if hasattr(sig, 'name') else str(sig)} a los procesos {pids} del puerto {port}.")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interfaz de red")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Puerto")] = 8000,
    out: Annotated[Path, typer.Option("--out", "-o", help="Directorio de salida de los audiolibros")] = Path("out"),
) -> None:
    """Arranca la interfaz web para subir EPUB y seguir el progreso."""
    from .web.server import serve as run_server

    log(f"Interfaz web en http://{host}:{port}")
    run_server(host=host, port=port, out_dir=out)


def _xtts_language(book_language: str | None) -> str:
    from .tts_xtts import SUPPORTED_LANGUAGES

    if not book_language:
        return "es"
    code = book_language.lower()
    if code.startswith("zh"):
        return "zh-cn"
    code = code.split("-")[0]
    return code if code in SUPPORTED_LANGUAGES else "es"


if __name__ == "__main__":
    app()
