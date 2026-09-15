"""Cola de trabajos en memoria con un único hilo de síntesis (la GPU solo admite un trabajo a la vez)."""
from __future__ import annotations

import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import pipeline
from ..models import Chapter
from ..pipeline import PipelineOptions
from ..tts_xtts import XttsEngine, loaded_model_devices, pick_device

MAX_LOG_LINES = 400
MAX_EVENTS = 300
ACTIVE_STATUSES = ("queued", "extracting", "synthesizing", "packing")
TERMINAL_STATUSES = ("done", "error", "cancelled")
_RICH_MARKUP = re.compile(r"\[/?[a-z ]+\]")


class JobCancelled(Exception):
    pass


@dataclass
class JobOptions:
    speaker: str | None = None
    speaker_wav: str | None = None
    language: str | None = None
    speed: float = 1.0
    min_words: int = 100
    toc_depth: int = 1
    chapters: str | None = None
    device: str = "auto"
    audiobook: bool = True


@dataclass
class ChapterState:
    index: int
    title: str
    words: int
    status: str = "pending"  # pending | running | done | skipped
    audio_file: str | None = None


@dataclass
class Job:
    id: str
    filename: str
    epub_path: Path
    out_dir: Path
    options: JobOptions
    status: str = "queued"  # queued | extracting | synthesizing | packing | done | error | cancelled
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    title: str | None = None
    author: str | None = None
    cover: str | None = None
    chapters: list[ChapterState] = field(default_factory=list)
    current_chapter: int | None = None
    fragment_done: int = 0
    fragment_total: int = 0
    audiobook: str | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    book_dir: Path | None = None
    cancel_requested: bool = False
    pause_requested: bool = False  # el worker se detiene al acabar el fragmento en curso
    paused: bool = False  # True mientras el worker está realmente detenido en este trabajo

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("epub_path")
        data.pop("out_dir")
        data.pop("cancel_requested")
        data["book_dir"] = str(self.book_dir) if self.book_dir else None
        done = sum(1 for c in self.chapters if c.status in ("done", "skipped"))
        data["chapters_done"] = done
        data["progress"] = _overall_progress(self, done)
        return data


def _overall_progress(job: Job, done: int) -> float:
    if job.status == "done":
        return 1.0
    if job.status in ("queued", "extracting") or not job.chapters:
        return 0.0
    selected = [c for c in job.chapters if c.status != "pending" or _is_selected(job, c.index)]
    total = max(1, len(selected))
    partial = (job.fragment_done / job.fragment_total) if job.fragment_total else 0.0
    return min(0.99, (done + partial) / total)


def _is_selected(job: Job, index: int) -> bool:
    ranges = job.options.chapters
    if not ranges:
        return True
    from ..cli import parse_ranges

    try:
        selected = parse_ranges(ranges)
    except Exception:
        return True
    return selected is None or index in selected


class JobManager:
    def __init__(self, uploads_dir: Path, out_dir: Path) -> None:
        self.uploads_dir = uploads_dir
        self.out_dir = out_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._events: deque[str] = deque(maxlen=MAX_EVENTS)
        self._started_at = time.time()
        self._current_job_id: str | None = None
        self._worker_state = "starting"  # starting | idle | busy | paused | dead
        self._processed = 0
        self._worker = threading.Thread(target=self._run, name="xtts-worker", daemon=True)
        self._worker.start()
        self.event(f"Servidor iniciado. Salida: {self.out_dir.resolve()}")

    # ------------------------------------------------------------------ API
    def submit(self, filename: str, data: bytes, options: JobOptions) -> Job:
        job_id = uuid.uuid4().hex[:10]
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "libro.epub"
        epub_path = self.uploads_dir / f"{job_id}-{safe_name}"
        epub_path.write_bytes(data)
        job = Job(id=job_id, filename=filename, epub_path=epub_path, out_dir=self.out_dir, options=options)
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        self.event(f"[{job_id}] En cola: {filename} ({len(data) // 1024} KB)")
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status in ("done", "error", "cancelled"):
            return False
        job.cancel_requested = True
        job.pause_requested = False  # un trabajo pausado también se puede cancelar
        self.event(f"[{job_id}] Cancelación solicitada (estado: {job.status})")
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = datetime.now(timezone.utc).isoformat()
        return True

    def pause(self, job_id: str) -> bool:
        """Pide pausar un trabajo activo. Surte efecto al terminar el fragmento XTTS en curso (segundos)."""
        job = self._jobs.get(job_id)
        if not job or job.status not in ACTIVE_STATUSES or job.cancel_requested:
            return False
        if not job.pause_requested:
            job.pause_requested = True
            self.event(f"[{job_id}] Pausa solicitada (estado: {job.status})")
        return True

    def resume(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or not job.pause_requested:
            return False
        job.pause_requested = False
        self.event(f"[{job_id}] Reanudado")
        return True

    def delete(self, job_id: str, force: bool = False) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if not force and job.status in ACTIVE_STATUSES:
            return False
        return self._drop_job(job_id, job)

    def delete_all(self, force: bool = False) -> int:
        removed = 0
        # Snapshot fuera del lock: _drop_job adquiere el lock por su cuenta (no es reentrante).
        for job_id, job in list(self._jobs.items()):
            if not force and job.status in ACTIVE_STATUSES:
                continue
            if self._drop_job(job_id, job):
                removed += 1
        return removed

    def purge_stale_terminal_jobs(self) -> int:
        """Remueve trabajos terminados de la memoria del proceso sin tocar el hilo del worker activo."""
        removed = 0
        for job_id, job in list(self._jobs.items()):
            if job.status in TERMINAL_STATUSES and self._drop_job(job_id, job):
                removed += 1
        return removed

    def _drop_job(self, job_id: str, job: Job) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            del self._jobs[job_id]
        self.event(f"[{job_id}] Eliminado de la lista ({job.status})")
        try:
            if job.epub_path.exists():
                job.epub_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if job.book_dir and job.book_dir.exists():
                shutil.rmtree(job.book_dir)
        except Exception:
            pass
        return True

    # --------------------------------------------------------------- estado
    def event(self, message: str) -> None:
        """Registro global del backend (independiente de cada trabajo), visible en /api/status."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self._events.append(f"{stamp} {_RICH_MARKUP.sub('', str(message))}")

    def status(self) -> dict:
        jobs = list(self._jobs.values())
        counts = {s: 0 for s in ACTIVE_STATUSES + TERMINAL_STATUSES}
        for job in jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        worker_alive = self._worker.is_alive()
        current = self._jobs.get(self._current_job_id) if self._current_job_id else None
        return {
            "uptime_s": round(time.time() - self._started_at),
            "worker": {
                "alive": worker_alive,
                "state": self._worker_state if worker_alive else "dead",
                "current_job": current.id if current else None,
                "current_title": (current.title or current.filename) if current else None,
                "queue_size": counts["queued"],
                "processed": self._processed,
            },
            "device": {
                "default": pick_device(None),
                "model_loaded_on": loaded_model_devices(),
            },
            "jobs": counts,
            "out_dir": str(self.out_dir.resolve()),
            "uploads_dir": str(self.uploads_dir.resolve()),
            "events": list(self._events),
        }

    # --------------------------------------------------------------- worker
    def _run(self) -> None:
        self._worker_state = "idle"
        while True:
            job_id = self._queue.get()
            job = self._jobs.get(job_id)
            if not job or job.status == "cancelled":
                continue
            self._current_job_id = job_id
            self._worker_state = "busy"
            self.event(f"[{job_id}] Empieza el procesado de {job.filename}")
            try:
                self._process(job)
                self.event(f"[{job_id}] Terminado: {job.title or job.filename}")
            except JobCancelled:
                job.status = "cancelled"
                self._log(job, "Trabajo cancelado por el usuario")
                self.event(f"[{job_id}] Cancelado por el usuario")
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                self._log(job, f"ERROR: {job.error}")
                self._log(job, traceback.format_exc()[-1500:])
                self.event(f"[{job_id}] ERROR: {job.error}")
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.current_chapter = None
                self._processed += 1
                self._current_job_id = None
                self._worker_state = "idle"

    def _process(self, job: Job) -> None:
        job.status = "extracting"
        job.started_at = datetime.now(timezone.utc).isoformat()
        log = lambda msg: self._log(job, msg)  # noqa: E731

        from ..cli import _xtts_language, parse_ranges

        opts = PipelineOptions(
            epub_path=job.epub_path,
            out_dir=job.out_dir,
            min_words=job.options.min_words,
            toc_depth=job.options.toc_depth,
            only=parse_ranges(job.options.chapters),
            audiobook=job.options.audiobook,
        )
        book, book_dir = pipeline.prepare(opts, log)
        job.book_dir = book_dir
        job.title, job.author = book.meta.title, book.meta.author
        job.cover = pipeline.cover_path_for(book, book_dir).name if book.cover else None
        job.chapters = [ChapterState(index=c.index, title=c.title, words=c.words) for c in book.chapters]
        if not book.chapters:
            raise RuntimeError("No se detectaron capítulos con el umbral de palabras indicado")
        self._check_cancel(job)

        job.status = "synthesizing"
        engine = XttsEngine(
            speaker=job.options.speaker or None,
            speaker_wav=job.options.speaker_wav or None,
            language=job.options.language or _xtts_language(book.meta.language),
            speed=job.options.speed,
            device=job.options.device,
            logger=log,
            cancel_check=lambda: self._check_control(job),
        )

        def on_progress(done: int, total: int) -> None:
            job.fragment_done, job.fragment_total = done, total
            self._check_control(job)

        def on_chapter(chapter: Chapter, state: str) -> None:
            st = job.chapters[chapter.index - 1]
            if state == "start":
                job.current_chapter = chapter.index
                job.fragment_done = job.fragment_total = 0
                st.status = "running"
            else:
                st.status = state
                st.audio_file = f"audio/{pipeline.chapter_basename(chapter, len(book.chapters))}.m4a"
                job.current_chapter = None

        audio_files = pipeline.synthesize(book, book_dir, engine, opts, log, on_progress, on_chapter)
        for c in job.chapters:
            if c.index in audio_files:
                c.audio_file = f"audio/{audio_files[c.index].name}"
                if c.status == "pending":
                    c.status = "skipped"
        self._check_cancel(job)

        audiobook = None
        if opts.audiobook:
            job.status = "packing"
            audiobook = pipeline.pack(book, book_dir, audio_files, log)
        job.audiobook = audiobook.name if audiobook else None
        pipeline.write_manifest(book, book_dir, opts, engine, audio_files, audiobook)
        job.status = "done"
        log("Terminado")

    def _check_cancel(self, job: Job) -> None:
        if job.cancel_requested:
            raise JobCancelled()

    def _check_control(self, job: Job, poll_s: float = 0.2) -> None:
        """Punto de control entre fragmentos: cancela si se pidió, o bloquea al worker mientras el trabajo esté pausado."""
        self._check_cancel(job)
        if not job.pause_requested:
            return
        job.paused = True
        self._worker_state = "paused"
        self._log(job, "En pausa (el trabajo se reanuda desde este fragmento)")
        self.event(f"[{job.id}] Worker en pausa")
        try:
            while job.pause_requested and not job.cancel_requested:
                time.sleep(poll_s)
        finally:
            job.paused = False
            self._worker_state = "busy"
        self._log(job, "Reanudado")
        self._check_cancel(job)

    def _log(self, job: Job, message: str) -> None:
        clean = _RICH_MARKUP.sub("", str(message))
        stamp = datetime.now().strftime("%H:%M:%S")
        job.log.append(f"{stamp} {clean}")
        if len(job.log) > MAX_LOG_LINES:
            del job.log[: len(job.log) - MAX_LOG_LINES]
        # Las trazas del motor (carga del modelo, fallback MPS→CPU, avisos) también van al registro global.
        if "XTTS" in clean or "MPS" in clean or "Modelo listo" in clean:
            self.event(f"[{job.id}] {clean}")
