"""Interfaz web: subir EPUB, seguir el progreso y descargar el audiolibro."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..tts_xtts import DEFAULT_SPEAKER, SUPPORTED_LANGUAGES, XTTS_SPEAKERS
from .jobs import JobManager, JobOptions

STATIC = Path(__file__).parent / "static"


def create_app(out_dir: Path = Path("out"), uploads_dir: Path | None = None) -> FastAPI:
    manager = JobManager(uploads_dir or out_dir / "_uploads", out_dir)
    app = FastAPI(title="ebook-generator", docs_url="/api/docs", redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config() -> dict:
        return {
            "speakers": XTTS_SPEAKERS,
            "default_speaker": DEFAULT_SPEAKER,
            "languages": sorted(SUPPORTED_LANGUAGES),
        }

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [j.to_dict() for j in manager.list()]

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
            device=device if device in ("auto", "mps", "cpu", "cuda") else "auto",
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

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        if not manager.delete(job_id):
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
