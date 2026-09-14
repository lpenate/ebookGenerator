"""Motor de síntesis XTTS v2 (Coqui) con aceleración Metal (MPS) en Apple Silicon."""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable

# Deben fijarse antes de importar torch / TTS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops sin kernel Metal caen a CPU
os.environ.setdefault("COQUI_TOS_AGREED", "1")  # acepta la Coqui Public Model License (no comercial)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .chunk import chunk_text  # noqa: E402

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24_000
DEFAULT_SPEAKER = "Alma María"
# Hablantes preentrenados incluidos en XTTS v2 (misma lista que devuelve el modelo).
XTTS_SPEAKERS = [
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence",
    "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler",
    "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black",
    "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid",
    "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando", "Lidiya Szekeres",
    "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
    "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer",
    "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı",
    "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski",
]
SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"}


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class XttsEngine:
    def __init__(
        self,
        speaker: str | None = None,
        speaker_wav: str | None = None,
        language: str = "es",
        speed: float = 1.0,
        device: str | None = None,
        pause_between_chunks: float = 0.35,
        pause_between_paragraphs: float = 0.6,
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"XTTS v2 no soporta el idioma '{language}'. Soportados: {sorted(SUPPORTED_LANGUAGES)}")
        self.language = language
        self.speed = speed
        self.speaker_wav = speaker_wav
        self.speaker = None if speaker_wav else (speaker or DEFAULT_SPEAKER)
        self.device = pick_device(device)
        self.pause_chunks = pause_between_chunks
        self.pause_paragraphs = pause_between_paragraphs
        self._tts = None

    # ------------------------------------------------------------------ carga
    def load(self) -> None:
        if self._tts is not None:
            return
        self._tts = _load_model(self.device)
        if self.speaker and self.speaker not in self.speakers():
            raise ValueError(f"Hablante desconocido '{self.speaker}'. Usa el comando `voices` para ver los disponibles.")

    def speakers(self) -> list[str]:
        self.load()
        return list(self._tts.synthesizer.tts_model.speaker_manager.speaker_names)

    # -------------------------------------------------------------- síntesis
    def synthesize_chunk(self, text: str) -> np.ndarray:
        self.load()
        kwargs = {"text": text, "language": self.language, "speed": self.speed, "split_sentences": True}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        else:
            kwargs["speaker"] = self.speaker
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav = self._tts.tts(**kwargs)
        audio = np.asarray(wav, dtype=np.float32)
        return _trim_silence(audio)

    def synthesize_text(self, text: str, on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Sintetiza un texto largo (un capítulo) y devuelve la onda completa a 24 kHz mono."""
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        chunks: list[tuple[str, bool]] = []  # (texto, es_fin_de_párrafo)
        for paragraph in paragraphs:
            pieces = chunk_text(paragraph)
            for i, piece in enumerate(pieces):
                chunks.append((piece, i == len(pieces) - 1))

        parts: list[np.ndarray] = []
        for i, (piece, ends_paragraph) in enumerate(chunks):
            parts.append(self.synthesize_chunk(piece))
            pause = self.pause_paragraphs if ends_paragraph else self.pause_chunks
            parts.append(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
            if on_progress:
                on_progress(i + 1, len(chunks))
        if not parts:
            return np.zeros(SAMPLE_RATE, dtype=np.float32)
        return np.concatenate(parts)


_MODEL_CACHE: dict[str, object] = {}


def _load_model(device: str):
    """Carga XTTS una sola vez por proceso y dispositivo (tarda varios segundos y ocupa ~2 GB)."""
    if device not in _MODEL_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from TTS.api import TTS

            tts = TTS(MODEL_NAME)
            tts.to(device)
        _MODEL_CACHE[device] = tts
    return _MODEL_CACHE[device]


def _trim_silence(audio: np.ndarray, threshold: float = 0.01, keep: int = 1200) -> np.ndarray:
    """Recorta silencio sobrante al principio y al final, dejando un pequeño margen."""
    above = np.flatnonzero(np.abs(audio) > threshold)
    if above.size == 0:
        return audio
    start = max(0, int(above[0]) - keep)
    end = min(audio.size, int(above[-1]) + keep)
    return audio[start:end]
