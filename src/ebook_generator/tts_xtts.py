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
# PyTorch (comprobado en 2.14) rechaza conv1d en MPS con entradas de más de 65536 muestras:
# «Output channels > 65536 not supported at the MPS device». El vocoder HiFiGAN de XTTS trabaja a 24 kHz,
# así que cualquier frase de más de ~2,7 s de audio rompe la síntesis en la GPU.
MPS_CONV1D_MAX_LENGTH = 65536
# Modos de dispositivo. `m1` es el modo híbrido para Apple Silicon: GPT en MPS y vocoder HiFiGAN en CPU,
# que esquiva el límite anterior sin renunciar a la GPU. Los demás modos se comportan como siempre.
DEVICE_CHOICES = ("auto", "m1", "mps", "cpu", "cuda")
M1_LABEL = "m1 (gpt en mps, vocoder en cpu)"


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        if requested not in DEVICE_CHOICES and not requested.startswith("cuda"):
            raise ValueError(f"Dispositivo desconocido '{requested}'. Opciones: {', '.join(DEVICE_CHOICES)}")
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
        logger: Callable[[str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
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
        self.logger = logger
        self.cancel_check = cancel_check
        self._tts = None

    # ------------------------------------------------------------------ carga
    def load(self) -> None:
        if self._tts is not None:
            return
        if self.cancel_check:
            self.cancel_check()
        self._tts = _load_model(self.device, logger=self.logger)
        if self.speaker and self.speaker not in self.speakers():
            raise ValueError(f"Hablante desconocido '{self.speaker}'. Usa el comando `voices` para ver los disponibles.")

    def speakers(self) -> list[str]:
        self.load()
        return list(self._tts.synthesizer.tts_model.speaker_manager.speaker_names)

    @property
    def device_label(self) -> str:
        """Dispositivo real de ejecución, p. ej. `mps (vocoder en cpu)` cuando se aplica el workaround de MPS."""
        label = getattr(self._tts, "ebook_device_label", None)
        return label if isinstance(label, str) else self.device

    # -------------------------------------------------------------- síntesis
    def synthesize_chunk(self, text: str) -> np.ndarray:
        if self.cancel_check:
            self.cancel_check()
        self.load()
        kwargs = {"text": text, "language": self.language, "speed": self.speed, "split_sentences": True}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        else:
            kwargs["speaker"] = self.speaker
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                wav = self._tts.tts(**kwargs)
            except Exception as exc:
                message = str(exc)
                if self.device in ("mps", "m1") and "not supported at the MPS device" in message:
                    text = "MPS rechazó el chunk XTTS durante la síntesis; reintento en CPU para recuperar la síntesis."
                    if self.logger:
                        self.logger(text)
                    warnings.warn(text, RuntimeWarning, stacklevel=2)
                    self.device = "cpu"
                    self._tts = _load_model("cpu", logger=self.logger)
                    self._tts.to("cpu")
                    wav = self._tts.tts(**kwargs)
                else:
                    raise
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
            if self.cancel_check:
                self.cancel_check()
            parts.append(self.synthesize_chunk(piece))
            pause = self.pause_paragraphs if ends_paragraph else self.pause_chunks
            parts.append(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
            if on_progress:
                on_progress(i + 1, len(chunks))
        if not parts:
            return np.zeros(SAMPLE_RATE, dtype=np.float32)
        return np.concatenate(parts)


_MODEL_CACHE: dict[str, object] = {}


def loaded_model_devices() -> list[str]:
    """Dispositivos reales sobre los que hay un modelo XTTS cargado en este proceso (vacío si aún no se cargó).

    Tras un fallback MPS→CPU la caché guarda el mismo modelo bajo ambas claves, así que se consulta
    el dispositivo de los pesos y no la clave.
    """
    devices: set[str] = set()
    for key, model in _MODEL_CACHE.items():
        label = getattr(model, "ebook_device_label", None)
        if isinstance(label, str):
            devices.add(label)
            continue
        real = key
        try:
            real = next(model.synthesizer.tts_model.parameters()).device.type
        except Exception:  # noqa: BLE001 — mocks o modelos sin parámetros accesibles
            pass
        devices.add(real)
    return sorted(devices)


def recommended_device() -> str:
    """Dispositivo sugerido para la UI: `m1` si MPS existe pero conv1d está limitado, si no el de `auto`."""
    device = pick_device(None)
    if device == "mps" and mps_conv1d_length_limited():
        return "m1"
    return device


def mps_conv1d_length_limited() -> bool:
    """Sondea (en milisegundos) si este PyTorch rechaza conv1d en MPS por encima de MPS_CONV1D_MAX_LENGTH muestras."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.zeros(1, 1, MPS_CONV1D_MAX_LENGTH + 1, device="mps")
        w = torch.zeros(1, 1, 3, device="mps")
        torch.nn.functional.conv1d(x, w, padding=1)
        return False
    except Exception:  # noqa: BLE001 — NotImplementedError en los PyTorch afectados; ante la duda, CPU (siempre correcto)
        return True


def _move_vocoder_to_cpu(tts) -> None:
    """Workaround del límite de conv1d en MPS: el vocoder HiFiGAN se ejecuta en CPU y el GPT sigue en MPS.

    El GPT autorregresivo es la parte cara de XTTS y no tiene el problema; el vocoder es barato en CPU.
    Medido en un M-series: ~2x más rápido que todo el modelo en CPU, con salida idéntica.
    """
    decoder = tts.synthesizer.tts_model.hifigan_decoder.waveform_decoder
    decoder.to("cpu")
    original = decoder.forward

    def forward(*args, **kwargs):
        device = next((a.device for a in [*args, *kwargs.values()] if torch.is_tensor(a)), None)
        args = [a.to("cpu") if torch.is_tensor(a) else a for a in args]
        kwargs = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        out = original(*args, **kwargs)
        return out.to(device) if device is not None and torch.is_tensor(out) else out

    decoder.forward = forward
    tts.ebook_device_label = M1_LABEL


def _load_model(device: str, logger: Callable[[str], None] | None = None):
    """Carga XTTS una sola vez por proceso y dispositivo.

    Si el backend MPS de Apple falla con el mensaje
    `Output channels > 65536 not supported at the MPS device`,
    el motor se reintenta sobre CPU para evitar romper la síntesis.
    """
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]
    if device == "cpu" and "cpu" in _MODEL_CACHE:
        return _MODEL_CACHE["cpu"]
    if device == "mps" and "cpu" in _MODEL_CACHE:
        return _MODEL_CACHE["cpu"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from TTS.api import TTS

        if device == "m1":
            if not torch.backends.mps.is_available():
                raise RuntimeError("El modo m1 necesita Apple Silicon con MPS disponible; usa --device cpu o auto.")
            tts = TTS(MODEL_NAME)
            tts.to("mps")
            _move_vocoder_to_cpu(tts)
            if logger:
                logger(
                    f"Modo m1: GPT de XTTS en MPS y vocoder HiFiGAN en CPU "
                    f"(PyTorch {torch.__version__} limita conv1d en MPS a {MPS_CONV1D_MAX_LENGTH} muestras)."
                )
            _MODEL_CACHE["m1"] = tts
            return tts

        try:
            tts = TTS(MODEL_NAME)
            tts.to(device)
            tts.ebook_device_label = device
        except Exception as exc:
            message = str(exc)
            if device == "mps" and "not supported at the MPS device" in message:
                text = "MPS rechazó el modelo XTTS; reintento en CPU para recuperar la síntesis."
                if logger:
                    logger(text)
                warnings.warn(text, RuntimeWarning, stacklevel=2)
                tts = TTS(MODEL_NAME)
                tts.to("cpu")
                tts.ebook_device_label = "cpu"
                _MODEL_CACHE["cpu"] = tts
                _MODEL_CACHE["mps"] = tts
                return tts
            raise

    _MODEL_CACHE[device] = tts
    return tts


def _trim_silence(audio: np.ndarray, threshold: float = 0.01, keep: int = 1200) -> np.ndarray:
    """Recorta silencio sobrante al principio y al final, dejando un pequeño margen."""
    above = np.flatnonzero(np.abs(audio) > threshold)
    if above.size == 0:
        return audio
    start = max(0, int(above[0]) - keep)
    end = min(audio.size, int(above[-1]) + keep)
    return audio[start:end]
