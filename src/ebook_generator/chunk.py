"""Troceado de texto para XTTS v2.

XTTS genera frase a frase y avisa/trunca por encima de ~400 tokens; para español el límite
práctico está en torno a 230 caracteres por frase. Agrupamos frases en bloques para reducir
el número de llamadas y cortamos las frases demasiado largas por comas o espacios.
"""
from __future__ import annotations

import re

MAX_SENTENCE_CHARS = 220
MAX_CHUNK_CHARS = 700

_SENTENCE_END = re.compile(r"(?<=[.!?…»”\"])\s+")
_SOFT_BREAK = re.compile(r"(?<=[,;:—–])\s+")


def split_sentences(paragraph: str, max_chars: int = MAX_SENTENCE_CHARS) -> list[str]:
    out: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        out.extend(_split_long(sentence, max_chars))
    return out


def _split_long(sentence: str, max_chars: int) -> list[str]:
    pieces = _SOFT_BREAK.split(sentence)
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            out.append(current)
        current = piece
        while len(current) > max_chars:
            cut = current.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(current[:cut].strip())
            current = current[cut:].strip()
    if current:
        out.append(current)
    return out


def chunk_text(text: str, max_chunk_chars: int = MAX_CHUNK_CHARS, max_sentence_chars: int = MAX_SENTENCE_CHARS) -> list[str]:
    """Devuelve bloques de frases completas, sin superar max_chunk_chars, respetando párrafos."""
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        sentences = split_sentences(paragraph, max_sentence_chars)
        if not sentences:
            continue
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= max_chunk_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        # Cierre de párrafo: forzamos corte para que la pausa entre párrafos sea natural.
        if current:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks
