from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BookMeta:
    title: str
    author: str | None = None
    language: str | None = None


@dataclass
class Chapter:
    index: int  # 1-based tras filtrar
    title: str
    href: str
    text: str
    words: int


@dataclass
class Cover:
    data: bytes
    media_type: str

    @property
    def extension(self) -> str:
        return {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}.get(
            self.media_type, "img"
        )


@dataclass
class Book:
    meta: BookMeta
    chapters: list[Chapter]
    skipped: list[tuple[str, str, int]] = field(default_factory=list)  # (href, title, words)
    cover: Cover | None = None
