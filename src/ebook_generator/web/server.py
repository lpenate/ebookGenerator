"""Interfaz web: subir EPUB, seguir el progreso y descargar el audiolibro."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..tts_xtts import DEFAULT_SPEAKER, DEVICE_CHOICES, SUPPORTED_LANGUAGES, XTTS_SPEAKERS, recommended_device
from .jobs import JobManager, JobOptions

STATIC = Path(__file__).parent / "static"


def create_app(out_dir: Path = Path("out"), uploads_dir: Path | None = None) -> FastAPI:
    manager = JobManager(uploads_dir or out_dir / "_uploads", out_dir)
    app = FastAPI(title="ebook-generator", docs_url="/api/docs", redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/i18n/{lang}.json")
    def translation(lang: str) -> JSONResponse:
        if lang not in {"es", "en"}:
            raise HTTPException(404, "Idioma no soportado")
        path = STATIC / "i18n" / f"{lang}.json"
        if not path.exists():
            raise HTTPException(404, "Fichero de traducción no encontrado")
        data = json.loads(path.read_text(encoding="utf-8"))
        return JSONResponse(data)

    @app.get("/api/config")
    def config() -> dict:
        return {
            "speakers": XTTS_SPEAKERS,
            "default_speaker": DEFAULT_SPEAKER,
            "languages": sorted(SUPPORTED_LANGUAGES),
            "devices": list(DEVICE_CHOICES),
            "recommended_device": recommended_device(),
        }

    @app.get("/api/status")
    def status() -> dict:
        """Estado interno del backend: worker, cola, dispositivo, modelo cargado y registro global."""
        return manager.status()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [j.to_dict() for j in manager.list()]

    @app.post("/api/jobs/purge")
    def purge_jobs(force: bool = False) -> dict:
        removed = manager.purge_stale_terminal_jobs() if not force else manager.delete_all(force=True)
        return {"ok": True, "removed": removed}

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        file: Annotated[UploadFile, File()],
        speaker: Annotated[str, Form()] = "",
        language: Annotated[str, Form()] = "",
        speed: Annotated[float, Form()] = 1.0,
        min_words: Annotated[int, Form()] = 100,
        toc_depth: Annotated[int, Form()] = 1,
        chapters: Annotated[str, Form()] = "",
        audiobook: Annotated[bool, Form()] = True,
        device: Annotated[str, Form()] = "auto",
    ) -> dict:
        name = file.filename or "libro.epub"
        if not name.lower().endswith(".epub"):
            raise HTTPException(400, "Solo se admiten ficheros .epub")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Fichero vacío")
        if speaker and speaker not in XTTS_SPEAKERS:
            raise HTTPException(400, f"Hablante desconocido: {speaker}")
        if language and language not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, f"Idioma no soportado: {language}")
        options = JobOptions(
            speaker=speaker or None,
            language=language or None,
            speed=max(0.5, min(2.0, speed)),
            min_words=max(0, min_words),
            toc_depth=max(1, min(4, toc_depth)),
            chapters=chapters.strip() or None,
            audiobook=audiobook,
            device=device if device in DEVICE_CHOICES else "auto",
        )
        job = manager.submit(name, data, options)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        if not manager.cancel(job_id):
            raise HTTPException(409, "No se puede cancelar este trabajo")
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/pause")
    def pause_job(job_id: str) -> dict:
        if not manager.pause(job_id):
            raise HTTPException(409, "Solo se puede pausar un trabajo activo")
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict:
        if not manager.resume(job_id):
            raise HTTPException(409, "El trabajo no está pausado")
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, force: bool = False) -> dict:
        if not manager.delete(job_id, force=force):
            raise HTTPException(409, "No se puede borrar un trabajo en curso")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str) -> FileResponse:
        job = manager.get(job_id)
        if not job or not job.book_dir:
            raise HTTPException(404, "Trabajo no encontrado")
        root = job.book_dir.resolve()
        target = (root / path).resolve()
        if root not in target.parents and target != root:
            raise HTTPException(403, "Ruta no permitida")
        if not target.is_file():
            raise HTTPException(404, "Fichero no encontrado")
        return FileResponse(target, filename=target.name)

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception) -> JSONResponse:  # noqa: ANN001
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, out_dir: Path = Path("out")) -> None:
    import uvicorn

    uvicorn.run(create_app(out_dir=out_dir), host=host, port=port, log_level="info")
