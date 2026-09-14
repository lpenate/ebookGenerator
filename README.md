# ebook-generator

Convierte un EPUB en un audiolibro completo, en local y con la GPU del Mac (Metal), usando **XTTS v2** en español:

1. **Extrae los capítulos** a texto plano (un `.txt` por capítulo), en el orden del *spine* y con los títulos de la tabla de contenidos (EPUB 2 y 3).
2. **Extrae la portada** del EPUB.
3. **Genera una locución** de cada capítulo con XTTS v2 y guarda **un audio por capítulo** (`.m4a`, AAC).
4. **Empaqueta un audiolibro `.m4b`** con la estructura estándar: marcadores de capítulo, carátula, título, autor y género *Audiobook*. Se abre en Apple Books, Audiobookshelf, BookPlayer, VLC, etc.

Incluye una **CLI** y una **interfaz web** para subir varios EPUB, verlos avanzar capítulo a capítulo y descargar el resultado.

---

## Requisitos

| Qué                       | Para qué                                              | Cómo se instala                              |
|---------------------------|-------------------------------------------------------|----------------------------------------------|
| macOS con Apple Silicon   | Aceleración Metal (MPS). En Intel o Linux funciona en CPU, mucho más lento. | —                          |
| 16 GB de RAM o más        | El modelo ocupa ~2 GB en memoria durante la síntesis  | —                                            |
| ~3 GB de disco            | Modelo XTTS v2 (1,9 GB) + PyTorch y dependencias      | —                                            |
| [Homebrew](https://brew.sh) | Instalar `uv` y `ffmpeg`                            | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| `uv`                      | Gestiona Python 3.12 y el entorno virtual             | `brew install uv`                            |
| `ffmpeg` (con `ffprobe`)  | Codificar AAC y montar el M4B                         | `brew install ffmpeg`                        |
| Python 3.12               | Lo descarga `uv` automáticamente; no hace falta instalarlo a mano | (automático con `uv sync`)      |
| Conexión a internet       | Solo la primera vez, para descargar el modelo         | —                                            |

No hace falta ninguna clave de API: todo se ejecuta en la máquina.

### Instalación paso a paso

```bash
brew install uv ffmpeg
```

```bash
git clone <este-repo> ebookGenerator && cd ebookGenerator
```

```bash
uv sync
```

`uv sync` crea `.venv/` con Python 3.12, PyTorch con soporte MPS, `coqui-tts` y el resto de dependencias. Tarda unos minutos la primera vez.

La primera ejecución de cualquier comando que sintetice audio descarga el modelo XTTS v2 (1,9 GB) a `~/Library/Application Support/tts/`. Comprueba que todo está en su sitio con una frase corta:

```bash
uv run ebook-generator sample -o out/prueba.wav
```

Debería mostrar `Modelo cargado en …s (mps)` y generar un WAV de unos 5 segundos.

### Cómo probarlo en cinco minutos

El repositorio incluye un EPUB de dominio público en `epub-sample/azul.epub` (*Azul...*, Rubén Darío, Project Gutenberg).

1. Comprueba la detección de capítulos y la portada, sin gastar tiempo de GPU:

   ```bash
   uv run ebook-generator extract --list epub-sample/azul.epub
   ```

2. Genera un solo relato corto (unos 220 palabras, menos de un minuto de síntesis) y escúchalo:

   ```bash
   uv run ebook-generator build epub-sample/azul.epub --chapters 15 --no-audiobook
   ```

   ```bash
   open out/azul-obras-completas-vol-iv/audio/
   ```

3. Prueba otro hablante o velocidad antes de comprometerte con el libro entero:

   ```bash
   uv run ebook-generator sample "La noche caía despacio sobre la ciudad." --speaker "Luis Moray" -o out/luis.wav
   ```

4. Genera el audiolibro completo (unas 4 horas de audio, 2 a 3 horas de proceso en un M5) y ábrelo en Apple Books:

   ```bash
   uv run ebook-generator build epub-sample/azul.epub --chapters 2-40
   ```

   ```bash
   open out/azul-obras-completas-vol-iv/*.m4b
   ```

   El rango `2-40` deja fuera la cabecera y la licencia de Project Gutenberg, que aparecen como capítulos 1 y 41 en la lista.

5. Lo mismo desde la interfaz web: `uv run ebook-generator serve`, abre <http://127.0.0.1:8000>, arrastra el EPUB y sigue el progreso.

---

## Uso por línea de comandos

Ver qué capítulos detecta antes de generar nada:

```bash
uv run ebook-generator extract --list libro.epub
```

Solo texto y portada:

```bash
uv run ebook-generator extract libro.epub
```

Todo: texto, portada, un audio por capítulo y M4B final:

```bash
uv run ebook-generator build libro.epub
```

Elegir hablante, velocidad y solo algunos capítulos:

```bash
uv run ebook-generator build libro.epub --speaker "Luis Moray" --speed 1.05 --chapters 1-3,7
```

Clonar una voz a partir de un WAV limpio de 6 a 30 segundos:

```bash
uv run ebook-generator build libro.epub --speaker-wav narrador.wav
```

Regenerar solo el M4B a partir de los audios ya generados:

```bash
uv run ebook-generator pack libro.epub
```

Listar los 58 hablantes preentrenados o probar uno:

```bash
uv run ebook-generator voices
```

```bash
uv run ebook-generator sample "Texto de prueba." --speaker "Alma María" -o out/alma.wav
```

Si un proceso se corta, vuelve a lanzar el mismo `build`: los capítulos con audio ya generado se saltan y continúa por donde estaba. Usa `--force` para regenerarlos.

### Opciones de `build`

| Opción                   | Descripción                                                           | Por defecto            |
|--------------------------|-----------------------------------------------------------------------|------------------------|
| `-o, --out <dir>`        | Directorio de salida                                                  | `out`                  |
| `-s, --speaker <nombre>` | Hablante preentrenado de XTTS v2                                      | `Alma María`           |
| `--speaker-wav <wav>`    | WAV de referencia para clonar una voz (anula `--speaker`)             | —                      |
| `-l, --language <código>`| Idioma XTTS (`es`, `en`, `fr`, `de`, `it`, `pt`, …)                   | el del EPUB, o `es`    |
| `--speed <n>`            | Velocidad relativa                                                    | `1.0`                  |
| `--device <d>`           | `auto`, `mps`, `cpu`, `cuda`                                          | `auto` (MPS en Mac)    |
| `-c, --chapters <r>`     | Capítulos a generar, p. ej. `1-3,7`                                   | todos                  |
| `--min-words <n>`        | Umbral para descartar portada, créditos, dedicatorias…                | `100`                  |
| `--toc-depth <n>`        | Nivel del índice que define un capítulo (2 para libros "Parte > Capítulo") | `1`               |
| `--bitrate <b>`          | Bitrate AAC por capítulo                                              | `64k`                  |
| `-f, --force`            | Regenerar audios ya existentes                                        | no                     |
| `--no-audiobook`         | No generar el M4B                                                     | —                      |
| `--keep-wav`             | Conservar los WAV intermedios                                         | no                     |

---

## Interfaz web

```bash
uv run ebook-generator serve
```

Abre <http://127.0.0.1:8000>. Desde ahí puedes:

- Arrastrar uno o varios EPUB y elegir hablante, idioma, velocidad, umbral de palabras y capítulos.
- Ver la cola de trabajos, con progreso global, capítulo en curso y fragmento en curso.
- Ver la portada extraída y el registro de cada trabajo en tiempo real.
- Descargar el M4B, cada capítulo `.m4a` y el `book.json`.
- Cancelar un trabajo en curso.

Los trabajos se procesan de uno en uno (la GPU solo admite una síntesis a la vez) y el modelo se carga una única vez por sesión del servidor. La cola vive en memoria: si reinicias el servidor desaparece la lista, pero los ficheros generados en `out/` se conservan y un `build` posterior los reutiliza.

Opciones: `--host 0.0.0.0` para acceder desde otros equipos de la red, `--port`, `--out`. La API REST está documentada en `/api/docs`.

---

## Salida

```
out/<titulo-del-libro>/
├── book.json                     # manifiesto: metadatos, hablante, capítulos, rutas, descartados
├── cover.jpg                     # portada extraída (jpg/png según el EPUB)
├── <titulo-del-libro>.m4b        # audiolibro con capítulos, carátula y metadatos
├── text/
│   ├── 01-uno-el-despertar.txt
│   └── 02-dos-la-huida.txt
└── audio/
    ├── 01-uno-el-despertar.m4a   # AAC mono 24 kHz, con título, número de pista, álbum y autor
    └── 02-dos-la-huida.m4a
```

---

## Rendimiento y calidad

- En un Apple M5 con 16 GB, XTTS v2 sobre MPS rinde entre **1,5 y 2,5 veces tiempo real**: una novela de 10 horas tarda entre 4 y 7 horas. Algunas operaciones del modelo no tienen kernel Metal y caen a CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`, que el programa fija solo).
- El texto se trocea en frases de hasta 220 caracteres agrupadas en bloques de 700, porque XTTS degrada la calidad por encima de ~400 tokens. Entre bloques se insertan pausas de 0,35 s y entre párrafos de 0,6 s.
- La **interpretación** en XTTS no se dirige con instrucciones de estilo: viene de la voz elegida. Prueba varios hablantes con `sample` o clona un narrador real con `--speaker-wav` para conseguir el tono que buscas. Para español suelen funcionar bien `Alma María`, `Luis Moray`, `Ferran Simen` y `Ana Florence`.

## Cómo se detectan los capítulos

- Se recorre el *spine* en orden, ignorando los elementos `linear="no"`.
- La tabla de contenidos (nav de EPUB 3 o NCX de EPUB 2) define los capítulos. Por defecto se usan las entradas de primer nivel; con `--toc-depth 2` también sus hijas.
- Si varias entradas del índice apuntan a un mismo fichero XHTML con anclas distintas (habitual en Project Gutenberg), el fichero se **divide en esas anclas**.
- El texto que aparece antes de la primera ancla de un fichero, o un fichero sin entrada en el índice y sin encabezado propio, se considera **continuación del capítulo anterior** y se fusiona con él.
- Sin índice, el título sale del primer `h1`–`h3`; si tampoco existe, `Capítulo N`.
- Se eliminan imágenes, enlaces (se conserva su texto) y `<sup>` (referencias a notas al pie); se quitan guiones blandos y espacios duros.
- Los bloques con menos de `--min-words` palabras se descartan y se listan al final de `--list`.

## Limitaciones conocidas

- Las cabeceras y licencias de Project Gutenberg aparecen como capítulos; exclúyelos con `--chapters`.
- XTTS v2 a veces alucina sílabas al final de una frase o lee mal cifras y abreviaturas poco comunes. Revisa el texto en `text/` y corrígelo antes de sintetizar si el libro es exigente.
- La cola de la interfaz web no persiste entre reinicios.

## Licencia del modelo

XTTS v2 se distribuye bajo la [Coqui Public Model License](https://coqui.ai/cpml), que **solo permite uso no comercial**. El programa acepta la licencia automáticamente (`COQUI_TOS_AGREED=1`). Si el uso es comercial hay que licenciar el modelo aparte o cambiar de motor.

## Estructura del código

```
src/ebook_generator/
├── cli.py            # comandos extract / build / pack / voices / sample / serve
├── pipeline.py       # orquestación: extraer -> texto + portada -> síntesis -> m4a -> m4b -> book.json
├── epub.py           # container.xml, OPF, spine, nav/NCX, portada, XHTML -> texto
├── chunk.py          # troceado por frases/bloques adaptado a XTTS
├── tts_xtts.py       # carga del modelo (MPS), hablantes, síntesis por fragmentos
├── audio.py          # WAV, ffmpeg: AAC por capítulo, M4B con capítulos y carátula
├── models.py
└── web/
    ├── server.py     # FastAPI: subida, estado, descargas
    ├── jobs.py       # cola en memoria con un hilo de síntesis
    └── static/index.html
```
