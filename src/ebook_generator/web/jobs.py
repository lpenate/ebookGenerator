"""Cola de trabajos en memoria con un único hilo de síntesis (la GPU solo admite un trabajo a la vez)."""
from __future__ import annotations

import queue
import re
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import pipeline
from ..models import Chapter
from ..pipeline import PipelineOptions
from ..tts_xtts import XttsEngine

MAX_LOG_LINES = 400
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
        self._worker = threading.Thread(target=self._run, name="xtts-worker", daemon=True)
        self._worker.start()

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
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = datetime.now(timezone.utc).isoformat()
        return True

    def delete(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status in ("synthesizing", "extracting", "packing"):
            return False
        with self._lock:
            self._jobs.pop(job_id, None)
        job.epub_path.unlink(missing_ok=True)
        return True

    # --------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self._jobs.get(job_id)
            if not job or job.status == "cancelled":
                continue
            try:
                self._process(job)
            except JobCancelled:
                job.status = "cancelled"
                self._log(job, "Trabajo cancelado por el usuario")
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                self._log(job, f"ERROR: {job.error}")
                self._log(job, traceback.format_exc()[-1500:])
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.current_chapter = None

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
        )

        def on_progress(done: int, total: int) -> None:
            job.fragment_done, job.fragment_total = done, total
            self._check_cancel(job)

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

    def _log(self, job: Job, message: str) -> None:
        clean = _RICH_MARKUP.sub("", str(message))
        stamp = datetime.now().strftime("%H:%M:%S")
        job.log.append(f"{stamp} {clean}")
        if len(job.log) > MAX_LOG_LINES:
            del job.log[: len(job.log) - MAX_LOG_LINES]
