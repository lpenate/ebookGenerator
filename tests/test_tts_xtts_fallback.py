import unittest
from unittest.mock import patch, Mock

from ebook_generator.tts_xtts import _load_model


class XttsDeviceFallbackTests(unittest.TestCase):
    def test_load_model_falls_back_to_cpu_when_mps_is_unsupported(self):
        fake_tts = Mock()

        def to_side_effect(device: str):
            if device == "mps":
                raise NotImplementedError("Output channels > 65536 not supported at the MPS device")
            return None

        fake_tts.to.side_effect = to_side_effect

        with patch("TTS.api.TTS", side_effect=[fake_tts, fake_tts]):
            import ebook_generator.tts_xtts as tts_module
            tts_module._MODEL_CACHE.pop("mps", None)

            loaded = _load_model("mps")

        self.assertIs(loaded, fake_tts)
        self.assertEqual(fake_tts.to.call_args_list[0].args[0], "mps")
        self.assertEqual(fake_tts.to.call_args_list[1].args[0], "cpu")


if __name__ == "__main__":
    unittest.main()
