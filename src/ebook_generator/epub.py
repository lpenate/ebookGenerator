"""Lectura de EPUB (2 y 3) con zipfile + lxml + BeautifulSoup.

Recorre container.xml -> OPF -> spine, lee la tabla de contenidos (nav o NCX) y localiza la portada.
Cuando varios capítulos comparten un mismo fichero XHTML (típico en Project Gutenberg), el fichero
se divide en las anclas a las que apunta la tabla de contenidos.
"""
from __future__ import annotations

import posixpath
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from lxml import etree

from .models import Book, BookMeta, Chapter, Cover

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
}

SOFT_HYPHEN = "­"
NBSP = " "


class EpubError(Exception):
    pass


@dataclass
class TocEntry:
    href: str  # ruta completa dentro del zip, sin fragmento
    fragment: str | None
    title: str
    depth: int  # 1 = nivel superior


def extract_epub(path: str | Path, min_words: int = 100, toc_depth: int = 1) -> Book:
    """Extrae metadatos, portada y capítulos.

    toc_depth: profundidad de la tabla de contenidos que define un capítulo (1 = entradas de
    primer nivel; 2 = también sus hijas, útil en libros con "Parte > Capítulo").
    """
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        def read(name: str) -> bytes | None:
            for candidate in (name, unquote(name)):
                if candidate in names:
                    return zf.read(candidate)
            return None

        container = read("META-INF/container.xml")
        if container is None:
            raise EpubError("EPUB inválido: falta META-INF/container.xml")
        opf_path = etree.fromstring(container).xpath("string(//c:rootfile/@full-path)", namespaces=NS)
        if not opf_path:
            raise EpubError("EPUB inválido: container.xml no declara rootfile")
        opf_dir = posixpath.dirname(opf_path)

        def resolve(href: str, base: str = opf_dir) -> str:
            joined = posixpath.join(base, unquote(href)) if base else unquote(href)
            return posixpath.normpath(joined)

        opf_bytes = read(opf_path)
        if opf_bytes is None:
            raise EpubError(f"EPUB inválido: no existe {opf_path}")
        opf = etree.fromstring(opf_bytes)

        meta = BookMeta(
            title=(opf.xpath("string(//dc:title)", namespaces=NS) or Path(path).stem).strip(),
            author=(opf.xpath("string(//dc:creator)", namespaces=NS) or "").strip() or None,
            language=(opf.xpath("string(//dc:language)", namespaces=NS) or "").strip() or None,
        )

        manifest: dict[str, dict[str, str]] = {}
        for item in opf.xpath("//opf:manifest/opf:item", namespaces=NS):
            manifest[item.get("id", "")] = {
                "href": item.get("href", ""),
                "media_type": item.get("media-type", ""),
                "properties": item.get("properties", ""),
            }

        spine = [
            manifest[ref.get("idref")]
            for ref in opf.xpath("//opf:spine/opf:itemref", namespaces=NS)
            if ref.get("linear", "yes") != "no" and ref.get("idref") in manifest
        ]

        toc = _read_toc(opf, manifest, resolve, read)
        cover = _find_cover(opf, manifest, resolve, read)

        chapters: list[Chapter] = []
        skipped: list[tuple[str, str, int]] = []

        def add(title: str, href: str, text: str) -> None:
            words = count_words(text)
            if words < min_words:
                skipped.append((href, title, words))
                return
            index = len(chapters) + 1
            chapters.append(Chapter(index=index, title=title or f"Capítulo {index}", href=href, text=text, words=words))

        def continue_previous(text: str) -> bool:
            """Texto sin entrada en el índice ni encabezado propio: es la continuación del capítulo anterior."""
            if not chapters or not text:
                return False
            last = chapters[-1]
            last.text = f"{last.text}\n\n{text}".strip()
            last.words = count_words(last.text)
            return True

        selected = [e for e in toc if e.depth <= max(1, toc_depth)]
        for item in spine:
            if not re.search(r"html|xml", item["media_type"], re.I):
                continue
            full = resolve(item["href"])
            html = read(full)
            if html is None:
                continue
            entries = [e for e in selected if e.href == full]
            anchored = [e for e in entries if e.fragment]
            if len(entries) >= 2 and anchored:
                for title, segment, listed in _split_by_anchors(html, entries):
                    text = html_to_text(segment)
                    if not listed and _first_heading(segment) is None and continue_previous(text):
                        continue
                    add(title, full, text)
            else:
                heading = _first_heading(html)
                text = html_to_text(html)
                if not entries and toc and heading is None and continue_previous(text):
                    continue
                add((entries[0].title if entries else None) or heading or "", full, text)

    return Book(meta=meta, chapters=chapters, skipped=skipped, cover=cover)


# --------------------------------------------------------------------------- TOC
def _read_toc(opf, manifest, resolve, read) -> list[TocEntry]:
    nav = next((i for i in manifest.values() if "nav" in i["properties"].split()), None)
    if nav:
        nav_path = resolve(nav["href"])
        nav_bytes = read(nav_path)
        if nav_bytes:
            entries = _parse_nav(nav_bytes, posixpath.dirname(nav_path))
            if entries:
                return entries
    ncx_id = opf.xpath("string(//opf:spine/@toc)", namespaces=NS)
    ncx = manifest.get(ncx_id) or next((i for i in manifest.values() if i["media_type"] == "application/x-dtbncx+xml"), None)
    if ncx:
        ncx_path = resolve(ncx["href"])
        ncx_bytes = read(ncx_path)
        if ncx_bytes:
            return _parse_ncx(ncx_bytes, posixpath.dirname(ncx_path))
    return []


def _split_href(base: str, href: str) -> tuple[str, str | None]:
    path, _, fragment = unquote(href).partition("#")
    full = posixpath.normpath(posixpath.join(base, path)) if path else ""
    return full, (fragment or None)


def _parse_nav(nav_bytes: bytes, base: str) -> list[TocEntry]:
    soup = BeautifulSoup(nav_bytes, "lxml-xml")
    toc = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", attrs={"role": "doc-toc"}) or soup.find("nav")
    if toc is None:
        return []
    entries: list[TocEntry] = []
    last_href = ""

    def walk(ol, depth: int) -> None:
        nonlocal last_href
        for li in ol.find_all("li", recursive=False):
            a = li.find("a", href=True, recursive=False) or li.find(["a", "span"], recursive=False)
            if a is not None:
                title = " ".join(a.get_text(" ").split())
                href, fragment = _split_href(base, a.get("href", "")) if a.has_attr("href") else (last_href, None)
                href = href or last_href
                last_href = href
                if title and href:
                    entries.append(TocEntry(href=href, fragment=fragment, title=title, depth=depth))
            for child in li.find_all("ol", recursive=False):
                walk(child, depth + 1)

    for ol in toc.find_all("ol", recursive=False):
        walk(ol, 1)
    return entries


def _parse_ncx(ncx_bytes: bytes, base: str) -> list[TocEntry]:
    root = etree.fromstring(ncx_bytes)
    entries: list[TocEntry] = []

    def walk(points, depth: int) -> None:
        for point in points:
            label = " ".join(point.xpath("string(ncx:navLabel/ncx:text)", namespaces=NS).split())
            src = point.xpath("string(ncx:content/@src)", namespaces=NS)
            if src and label:
                href, fragment = _split_href(base, src)
                entries.append(TocEntry(href=href, fragment=fragment, title=label, depth=depth))
            walk(point.xpath("ncx:navPoint", namespaces=NS), depth + 1)

    walk(root.xpath("//ncx:navMap/ncx:navPoint", namespaces=NS), 1)
    return entries


# --------------------------------------------------------------- división por anclas
_BODY_RE = re.compile(rb"<body\b[^>]*>", re.I)


def _anchor_position(html: bytes, fragment: str) -> int | None:
    pattern = re.compile(rb"<[a-zA-Z][^>]*\s(?:id|name)\s*=\s*[\"']" + re.escape(fragment.encode("utf-8")) + rb"[\"']", re.I)
    m = pattern.search(html)
    return m.start() if m else None


def _split_by_anchors(html: bytes, entries: list[TocEntry]) -> list[tuple[str, bytes, bool]]:
    """Corta el HTML en los elementos con los id de la tabla de contenidos.

    Devuelve (título, fragmento_html, listado_en_indice) en orden de documento. El texto anterior
    a la primera ancla se devuelve como bloque propio no listado: suele ser la continuación del
    capítulo anterior cuando el EPUB parte un capítulo en varios ficheros.
    """
    body = _BODY_RE.search(html)
    body_start = body.end() if body else 0
    marks: list[tuple[int, str]] = []
    for entry in entries:
        pos = body_start if entry.fragment is None else _anchor_position(html, entry.fragment)
        if pos is not None:
            marks.append((pos, entry.title))
    marks.sort(key=lambda m: m[0])
    if not marks:
        return [(entries[0].title, html, True)]

    segments: list[tuple[str, bytes, bool]] = []
    first_pos = marks[0][0]
    if first_pos > body_start:
        prefix = html[body_start:first_pos]
        segments.append((_first_heading(prefix) or "Preliminares", prefix, False))
    for i, (pos, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(html)
        segments.append((title, html[pos:end], True))
    return segments


# -------------------------------------------------------------------------- portada
def _find_cover(opf, manifest, resolve, read) -> Cover | None:
    candidates: list[dict[str, str]] = []
    candidates += [i for i in manifest.values() if "cover-image" in i["properties"].split()]
    cover_id = opf.xpath("string(//opf:metadata/opf:meta[@name='cover']/@content)", namespaces=NS)
    if cover_id and cover_id in manifest:
        candidates.append(manifest[cover_id])
    candidates += [
        i for i in manifest.values()
        if i["media_type"].startswith("image/") and re.search(r"cover|portada", i["href"], re.I)
    ]
    for item in candidates:
        if not item["media_type"].startswith("image/"):
            continue
        data = read(resolve(item["href"]))
        if data:
            return Cover(data=data, media_type=item["media_type"])
    return None


# ----------------------------------------------------------------------------- texto
def _first_heading(html: bytes) -> str | None:
    m = re.search(rb"<h[1-3]\b[^>]*>(.*?)</h[1-3]>", html, re.I | re.S)
    if not m:
        return None
    text = BeautifulSoup(m.group(1), "lxml").get_text(" ")
    return " ".join(text.split()) or None


BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "br", "tr", "section", "article",
    "aside", "figure", "figcaption", "dd", "dt", "pre", "hr", "table",
}
DROP_TAGS = {"img", "svg", "sup", "script", "style", "nav", "video", "audio", "math", "head", "title"}


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    for tag in body.find_all(DROP_TAGS):
        tag.decompose()
    for tag in body.find_all(BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    text = body.get_text("")
    text = text.replace(SOFT_HYPHEN, "").replace(NBSP, " ")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())
