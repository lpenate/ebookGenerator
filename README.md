# ebook-generator

Convert an EPUB into a complete local audiobook using the Mac GPU (Metal) and the Spanish XTTS v2 model:

1. Extract the chapters into plain text (one `.txt` file per chapter), following the EPUB spine order and preserving the table of contents titles (EPUB 2 and 3).
2. Extract the cover image from the EPUB.
3. Generate voice rendering for each chapter using XTTS v2 and save one audio file per chapter (`.m4a`, AAC).
4. Package the final audiobook as a `.m4b` using the standard structure: chapter markers, cover, title, author and `Audiobook` metadata. We recommend playing it with [BookPlayer](https://github.com/TortugaPower/BookPlayer); it also opens in Apple Books, Audiobookshelf, VLC, and similar readers.

The project includes both a CLI and a web interface for uploading multiple EPUBs, showing chapter-by-chapter progress, and downloading the result.

---

## Requirements

| What | Purpose | How to install |
|------|---------|----------------|
| macOS with Apple Silicon | Metal acceleration (MPS). On Intel or Linux it runs on CPU, which is much slower. | — |
| 16 GB RAM or more | The model uses roughly 2 GB of memory during synthesis | — |
| ~3 GB disk space | XTTS v2 model (1.9 GB) + PyTorch and dependencies | — |
| [Homebrew](https://brew.sh) | Install `uv` and `ffmpeg` | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| `uv` | Manages Python 3.12 and the virtual environment | `brew install uv` |
| `ffmpeg` (with `ffprobe`) | Encode AAC and package M4B | `brew install ffmpeg` |
| Python 3.12 | Downloaded automatically by `uv`; no need to install it manually | automatic via `uv sync` |
| Internet connection | Required only the first time to download the model | — |

No API key is required: everything runs locally on your machine.

### Step-by-step installation

```bash
brew install uv ffmpeg
```

```bash
git clone <this-repo> ebookGenerator && cd ebookGenerator
```

```bash
uv sync
```

`uv sync` creates `.venv/` with Python 3.12, PyTorch with MPS support, `coqui-tts`, and the rest of the dependencies. This takes a few minutes on the first run.

The first execution of any command that synthesizes audio downloads the XTTS v2 model (1.9 GB) into `~/Library/Application Support/tts/`. Verify that everything is in place with a short sentence:

```bash
uv run ebook-generator sample -o out/test.wav
```

It should show `Modelo cargado en …s (mps)` and generate a WAV file of about 5 seconds. On Apple Silicon, also try `--device m1` (see [Device modes](#device-modes)): it is the mode that actually keeps long sentences on the GPU with current PyTorch releases.

### How to try it in five minutes

The repository includes a public-domain EPUB in `epub-sample/azul.epub` (`Azul...`, Rubén Darío, Project Gutenberg).

1. Check chapter detection and cover extraction without spending GPU time:

   ```bash
   uv run ebook-generator extract --list epub-sample/azul.epub
   ```

2. Generate a single short passage (about 220 words, less than a minute of synthesis) and listen to it:

   ```bash
   uv run ebook-generator build epub-sample/azul.epub --chapters 15 --no-audiobook
   ```

   ```bash
   open out/azul-obras-completas-vol-iv/audio/
   ```

3. Try another speaker or speed before committing to the full book:

   ```bash
   uv run ebook-generator sample "La noche caía despacio sobre la ciudad." --speaker "Luis Moray" -o out/luis.wav
   ```

4. Generate the full audiobook (about 4 hours of audio, 2 to 3 hours of processing on an M5) and open it with BookPlayer or Apple Books:

   ```bash
   uv run ebook-generator build epub-sample/azul.epub --chapters 2-40
   ```

   ```bash
   open out/azul-obras-completas-vol-iv/*.m4b
   ```

   The `2-40` range excludes the header and Project Gutenberg license, which appear as chapters 1 and 41 in the chapter list.

5. Do the same through the web interface: `uv run ebook-generator serve`, open <http://127.0.0.1:8000>, drag the EPUB, and follow the progress.

---

## Command-line usage

See which chapters are detected before generating anything:

```bash
uv run ebook-generator extract --list libro.epub
```

Text and cover only:

```bash
uv run ebook-generator extract libro.epub
```

Everything: text, cover, one audio file per chapter, and final M4B:

```bash
uv run ebook-generator build libro.epub
```

Choose a speaker, speed, and only some chapters:

```bash
uv run ebook-generator build libro.epub --speaker "Luis Moray" --speed 1.05 --chapters 1-3,7
```

Clone a voice from a clean WAV sample of 6 to 30 seconds:

```bash
uv run ebook-generator build libro.epub --speaker-wav narrador.wav
```

Regenerate only the M4B from already generated audio:

```bash
uv run ebook-generator pack libro.epub
```

List the 58 pre-trained speakers or try one:

```bash
uv run ebook-generator voices
```

```bash
uv run ebook-generator sample "Sample text." --speaker "Alma María" -o out/alma.wav
```

If a process stops, simply rerun the same `build`: chapters with audio already generated are skipped and the build resumes where it left off. Use `--force` to regenerate them.

### `build` options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --out <dir>` | Output directory | `out` |
| `-s, --speaker <name>` | Pretrained XTTS v2 speaker | `Alma María` |
| `--speaker-wav <wav>` | Reference WAV to clone a voice (overrides `--speaker`) | — |
| `-l, --language <code>` | XTTS language (`es`, `en`, `fr`, `de`, `it`, `pt`, …) | language from EPUB, or `es` |
| `--speed <n>` | Relative speed | `1.0` |
| `--device <d>` | `auto`, `m1`, `mps`, `cpu`, `cuda` | `auto` (MPS on Mac) |
| `-c, --chapters <r>` | Chapters to generate, e.g. `1-3,7` | all |
| `--min-words <n>` | Threshold for discarding cover, credits, dedications, etc. | `100` |
| `--toc-depth <n>` | Index level that determines a chapter (2 for books with `Part > Chapter`) | `1` |
| `--bitrate <b>` | AAC bitrate per chapter | `64k` |
| `-f, --force` | Regenerate existing audio | no |
| `--no-audiobook` | Do not generate M4B | — |
| `--keep-wav` | Keep intermediate WAV files | no |

---

## Web interface

```bash
uv run ebook-generator serve
```

Open <http://127.0.0.1:8000>. From there you can:

- Drag and drop one or more EPUB files and choose speaker, language, speed, word threshold, and chapters.
- View the job queue with global progress, the current chapter, and the current fragment.
- View the extracted cover and the backend trace for each job in real time.
- Download the M4B, each chapter `.m4a`, and the `book.json` file.
- Cancel an ongoing job.

Jobs are processed one at a time (the GPU accepts only one synthesis task at a time), and the model is loaded once per server session. A running job can be **paused** and **resumed** from its card (or `POST /api/jobs/{id}/pause` and `/resume`): the worker stops after the fragment it is synthesizing, which takes a few seconds, and while it is paused nothing else in the queue runs. Cancelling keeps the chapters already rendered in `out/`, so re-submitting the same EPUB resumes at chapter granularity. The queue lives in memory: if you restart the server, the queue disappears, but the files in `out/` are preserved and a later `build` can reuse them.

The "Backend status" panel at the top of the job list shows what the server is doing: whether the synthesis worker is idle or busy and on which job, queue length, the device in use, whether the XTTS model is loaded (and on CPU or MPS), uptime, and a global server log with model loading, MPS→CPU fallbacks, errors and job lifecycle events. If the page loses contact with the server the panel turns red and says so. The same data is available as JSON at `/api/status`.

Options: `--host 0.0.0.0` to access the service from devices on the network, `--port`, and `--out`. The REST API is documented at `/api/docs`.

---

## Output

```
out/<book-title>/
├── book.json                     # manifest: metadata, speaker, chapters, paths, discarded files
├── cover.jpg                     # extracted cover (jpg/png depending on the EPUB)
├── <book-title>.m4b              # audiobook with chapters, cover, and metadata
├── text/
│   ├── 01-one-the-awakening.txt
│   └── 02-two-the-flight.txt
└── audio/
    ├── 01-one-the-awakening.m4a   # AAC mono 24 kHz, with title, track number, album and author
    └── 02-two-the-flight.m4a
```

### Recommended player: BookPlayer

To listen to the generated audiobooks we recommend **[BookPlayer](https://github.com/TortugaPower/BookPlayer)** (iOS and macOS, free and open source). It respects the M4B structure exactly as generated: it navigates chapter bookmarks, shows the cover and metadata, remembers the reading position, allows speed adjustment, and supports a sleep timer. To pass the file to BookPlayer, import it from Files, iCloud Drive, or AirDrop.

Other alternatives that also read the M4B with chapters: Apple Books, Audiobookshelf, and VLC.

---

## Device modes

| Mode | What runs where | When to use it |
|------|-----------------|----------------|
| `auto` | MPS if available, otherwise CUDA, otherwise CPU | Default. Unchanged behaviour. |
| `m1` | XTTS GPT (the expensive autoregressive part) on MPS, HiFiGAN vocoder on CPU | **Recommended on Apple Silicon with PyTorch ≥ 2.1x.** Measured about 2x faster than `cpu`, identical output. |
| `mps` | Whole model on MPS | Only useful with a PyTorch whose MPS `conv1d` accepts long inputs; otherwise every sentence longer than ~2.7 s fails and the engine falls back to CPU. |
| `cpu` | Whole model on CPU | Intel Macs, Linux without GPU, or debugging. Slow. |
| `cuda` | Whole model on an NVIDIA GPU | Linux/Windows with CUDA. |

Why `m1` exists: current PyTorch builds (checked with 2.14) refuse `conv1d` on MPS when the input is longer than 65536 samples (`Output channels > 65536 not supported at the MPS device`). The XTTS vocoder works at 24 kHz, so any sentence over ~2.7 s of audio trips it. With `mps`/`auto` the engine catches the error and reloads the model on CPU, which is why long jobs appeared to hang. `m1` keeps the GPT on the GPU and runs only the cheap vocoder on CPU. The web interface preselects `m1` when the server detects this limitation (`/api/config` → `recommended_device`).

## Performance and quality

- On an Apple M5 with 16 GB, `--device m1` runs at roughly **0.5 to 1x real time** in our measurements (GPT on MPS, vocoder on CPU); `cpu` is about half that. Some operations of the model do not have a Metal kernel and fall back to CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`, which the program sets automatically).
- Text is chunked into sentences up to 220 characters and grouped into blocks up to 700 characters, because XTTS quality degrades above roughly 400 tokens. Pauses of 0.35 s are inserted between blocks, and 0.6 s between paragraphs.
- The XTTS **performance/interpretation** is not controlled by style instructions; it comes from the selected voice. Try several speakers with `sample` or clone a real narrator with `--speaker-wav` to get the tone you want. For Spanish, `Alma María`, `Luis Moray`, `Ferran Simen`, and `Ana Florence` usually work well.

## How chapters are detected

- The spine is traversed in order, ignoring elements with `linear="no"`.
- The table of contents (`nav` in EPUB 3, or `NCX` in EPUB 2) defines the chapters. By default, first-level entries are used; with `--toc-depth 2`, child sections are included as well.
- If several index entries point to the same XHTML file with different anchors (common in Project Gutenberg), the file is **split into those anchors**.
- The text that appears before the first anchor in a file, or a file without a table-of-contents entry and without its own heading, is considered a **continuation of the previous chapter** and merged with it.
- Without an index, the title is derived from the first `h1`–`h3`; if no heading exists, the fallback is `Chapter N`.
- Images are removed, links are kept as readable text, and `<sup>` tags are removed (footnote references). Soft hyphens and hard spaces are also removed.
- Blocks with fewer than `--min-words` words are discarded and listed at the end of the `--list` output.

## Known limitations

- Project Gutenberg headers and licenses appear as chapters; exclude them using `--chapters`.
- XTTS v2 sometimes hallucinates syllables at the end of a sentence or reads uncommon numbers and abbreviations poorly. Review the text in `text/` and correct it before synthesizing if the book is demanding.
- The web interface queue does not persist across restarts.
- With `--device mps` or `auto`, PyTorch's MPS `conv1d` limit makes the engine fall back to CPU on the first long sentence. Use `--device m1` on Apple Silicon.

## Model license

XTTS v2 is distributed under the [Coqui Public Model License](https://coqui.ai/cpml), which **only permits non-commercial use**. The program accepts the license automatically (`COQUI_TOS_AGREED=1`). If you need commercial use, you must license the model separately or switch to another engine.

## Code structure

```
src/ebook_generator/
├── cli.py            # commands extract / build / pack / voices / sample / serve
├── pipeline.py       # orchestration: extract -> text + cover -> synthesis -> m4a -> m4b -> book.json
├── epub.py           # container.xml, OPF, spine, nav/NCX, cover, XHTML -> text
├── chunk.py          # sentence/block chunking adapted to XTTS
├── tts_xtts.py       # model loading (MPS), speaker selection, fragment synthesis
├── audio.py          # WAV, ffmpeg: AAC per chapter, M4B with chapters and cover
├── models.py
└── web/
    ├── server.py     # FastAPI: upload, status, downloads
    ├── jobs.py       # in-memory queue with a synthesis worker thread
    └── static/index.html
```
