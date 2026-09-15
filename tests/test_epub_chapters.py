import tempfile
import unittest
import zipfile
from pathlib import Path

from ebook_generator.epub import extract_epub, humanize_chapter_title


def _xhtml(body: str, heading: str | None = None) -> bytes:
    inner = f"<h1>{heading}</h1>\n" if heading else ""
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title></title></head>
<body>
{inner}{body}
</body>
</html>
""".encode()


def _long_body(tag: str, words: int = 120) -> str:
    return f"<p>{' '.join(f'{tag}{i}' for i in range(words))}</p>"


def _write_epub(path: Path, files: dict[str, bytes], spine: list[str], nav: list[tuple[str, str]]) -> None:
    manifest_items = "\n".join(
        f'    <item id="{name}" href="Text/{name}" media-type="application/xhtml+xml"/>' for name in spine
    )
    itemrefs = "\n".join(f'    <itemref idref="{name}"/>' for name in spine)
    nav_points = "\n".join(
        f"""    <navPoint id="nav-{i}" playOrder="{i}">
      <navLabel><text>{title}</text></navLabel>
      <content src="Text/{href}"/>
    </navPoint>"""
        for i, (title, href) in enumerate(nav, start=1)
    )
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package version="2.0" unique-identifier="BookId" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Libro de prueba</dc:title>
    <dc:creator>Autora</dc:creator>
    <dc:language>es</dc:language>
    <dc:identifier id="BookId">urn:uuid:test</dc:identifier>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
{manifest_items}
  </manifest>
  <spine toc="ncx">
{itemrefs}
  </spine>
</package>
"""
    ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
{nav_points}
  </navMap>
</ncx>
"""
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/toc.ncx", ncx)
        for name, data in files.items():
            zf.writestr(f"OEBPS/Text/{name}", data)


class ChapterDetectionTests(unittest.TestCase):
    def test_humanize_numeric_title(self):
        self.assertEqual(humanize_chapter_title("1", 1), "Capítulo 1")
        self.assertEqual(humanize_chapter_title("Primera parte", 1), "Primera parte")
        self.assertEqual(humanize_chapter_title("", 3), "Capítulo 3")

    def test_part_title_pages_do_not_swallow_the_following_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            epub = Path(tmp) / "klara-like.epub"
            files = {
                "cover.xhtml": _xhtml("", heading=None),
                "sinopsis.xhtml": _xhtml(_long_body("sinopsis", 150)),
                "parte1.xhtml": _xhtml("", heading="Primera parte"),
                "cuerpo1.xhtml": _xhtml(_long_body("uno", 150)),
                "parte2.xhtml": _xhtml("", heading="Segunda parte"),
                "cuerpo2.xhtml": _xhtml(_long_body("dos", 150)),
            }
            _write_epub(
                epub,
                files,
                spine=list(files),
                nav=[
                    ("Cubierta", "cover.xhtml"),
                    ("Primera parte", "parte1.xhtml"),
                    ("Segunda parte", "parte2.xhtml"),
                ],
            )
            book = extract_epub(epub, min_words=100)
            self.assertEqual([c.title for c in book.chapters], ["Primera parte", "Segunda parte"])
            self.assertTrue(book.chapters[0].text.startswith("uno0"))
            self.assertTrue(book.chapters[1].text.startswith("dos0"))
            skipped_hrefs = [href for href, _, _ in book.skipped]
            self.assertIn("OEBPS/Text/sinopsis.xhtml", skipped_hrefs)

    def test_unheaded_file_still_continues_the_previous_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            epub = Path(tmp) / "gutenberg-like.epub"
            files = {
                "cap1.xhtml": _xhtml(_long_body("alpha", 120), heading="Uno"),
                "cap1b.xhtml": _xhtml(_long_body("beta", 120)),
            }
            _write_epub(epub, files, spine=list(files), nav=[("Uno", "cap1.xhtml")])
            book = extract_epub(epub, min_words=50)
            self.assertEqual(len(book.chapters), 1)
            self.assertEqual(book.chapters[0].title, "Uno")
            self.assertIn("alpha0", book.chapters[0].text)
            self.assertIn("beta0", book.chapters[0].text)

    def test_numeric_headings_become_capitulo_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            epub = Path(tmp) / "numeric.epub"
            files = {
                "1.xhtml": _xhtml(_long_body("a", 120), heading="1"),
                "2.xhtml": _xhtml(_long_body("b", 120), heading="2"),
            }
            _write_epub(
                epub,
                files,
                spine=list(files),
                nav=[("1", "1.xhtml"), ("2", "2.xhtml")],
            )
            book = extract_epub(epub, min_words=50)
            self.assertEqual([c.title for c in book.chapters], ["Capítulo 1", "Capítulo 2"])


if __name__ == "__main__":
    unittest.main()
