import signal
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, Mock

from lxml import etree

from ebook_generator.epub import _read_toc, NS
from ebook_generator.tts_xtts import _load_model
from ebook_generator.web.jobs import JobManager, JobOptions
import ebook_generator.cli as cli_module


class XttsDeviceFallbackTests(unittest.TestCase):
    def test_load_model_falls_back_to_cpu_when_mps_is_unsupported(self):
        fake_tts = Mock()

        def to_side_effect(device: str):
            if device == "mps":
                raise NotImplementedError("Output channels > 65536 not supported at the MPS device")
            return None

        fake_tts.to.side_effect = to_side_effect

        with patch("TTS.api.TTS", return_value=fake_tts):
            import ebook_generator.tts_xtts as tts_module
            tts_module._MODEL_CACHE.clear()

            loaded = _load_model("mps")

        self.assertIs(loaded, fake_tts)
        self.assertEqual(tts_module._MODEL_CACHE["mps"], fake_tts)
        self.assertEqual(tts_module._MODEL_CACHE["cpu"], fake_tts)
        self.assertEqual(fake_tts.to.call_args_list[0].args[0], "mps")
        self.assertEqual(fake_tts.to.call_args_list[1].args[0], "cpu")
        self.assertEqual(fake_tts.ebook_device_label, "cpu")
        tts_module._MODEL_CACHE.clear()

    def test_m1_mode_keeps_gpt_on_mps_and_moves_vocoder_to_cpu(self):
        import torch
        import ebook_generator.tts_xtts as tts_module

        fake_tts = Mock()
        decoder = fake_tts.synthesizer.tts_model.hifigan_decoder.waveform_decoder
        decoder.forward = Mock(return_value=torch.ones(1, 1, 4))
        logs = []

        with patch("TTS.api.TTS", return_value=fake_tts), patch.object(tts_module.torch.backends.mps, "is_available", return_value=True):
            tts_module._MODEL_CACHE.clear()
            loaded = _load_model("m1", logger=logs.append)

        try:
            self.assertIs(loaded, fake_tts)
            self.assertIs(tts_module._MODEL_CACHE["m1"], fake_tts)
            self.assertNotIn("mps", tts_module._MODEL_CACHE)
            self.assertEqual(fake_tts.to.call_args_list[0].args[0], "mps")
            decoder.to.assert_called_once_with("cpu")
            self.assertEqual(fake_tts.ebook_device_label, tts_module.M1_LABEL)
            self.assertTrue(any("Modo m1" in line for line in logs))
            # La envoltura mueve las entradas a CPU y devuelve la salida en el dispositivo original.
            out = decoder.forward(torch.zeros(1, 1024, 8), g=torch.zeros(1, 512, 1))
            self.assertEqual(out.device.type, "cpu")
            self.assertEqual(tts_module.loaded_model_devices(), [tts_module.M1_LABEL])
        finally:
            tts_module._MODEL_CACHE.clear()

    def test_mps_mode_is_unchanged_and_does_not_touch_the_vocoder(self):
        import ebook_generator.tts_xtts as tts_module

        fake_tts = Mock()
        decoder = fake_tts.synthesizer.tts_model.hifigan_decoder.waveform_decoder
        with patch("TTS.api.TTS", return_value=fake_tts):
            tts_module._MODEL_CACHE.clear()
            loaded = _load_model("mps")
        try:
            self.assertIs(loaded, fake_tts)
            fake_tts.to.assert_called_once_with("mps")
            decoder.to.assert_not_called()
            self.assertEqual(fake_tts.ebook_device_label, "mps")
        finally:
            tts_module._MODEL_CACHE.clear()

    def test_m1_mode_requires_mps(self):
        import ebook_generator.tts_xtts as tts_module

        with patch("TTS.api.TTS", return_value=Mock()), patch.object(tts_module.torch.backends.mps, "is_available", return_value=False):
            tts_module._MODEL_CACHE.clear()
            with self.assertRaises(RuntimeError):
                _load_model("m1")

    def test_pick_device_accepts_m1_and_rejects_unknown_devices(self):
        from ebook_generator.tts_xtts import pick_device

        self.assertEqual(pick_device("m1"), "m1")
        self.assertEqual(pick_device("cpu"), "cpu")
        with self.assertRaises(ValueError):
            pick_device("gpu")

    def test_job_manager_force_delete_can_remove_active_stale_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uploads_dir = root / "uploads"
            out_dir = root / "out"
            manager = JobManager(uploads_dir, out_dir)
            job = manager.submit("libro.epub", b"dummy", JobOptions())
            job.status = "extracting"

            self.assertTrue(manager.delete(job.id, force=True))
            self.assertIsNone(manager.get(job.id))

    def test_job_manager_bulk_delete_and_purge_do_not_deadlock(self):
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = JobManager(root / "uploads", root / "out")
            done = manager.submit("a.epub", b"dummy", JobOptions())
            done.status = "done"
            running = manager.submit("b.epub", b"dummy", JobOptions())
            running.status = "synthesizing"
            failed = manager.submit("c.epub", b"dummy", JobOptions())
            failed.status = "error"

            result = {}

            def run():
                result["purged"] = manager.purge_stale_terminal_jobs()
                result["forced"] = manager.delete_all(force=True)

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive(), "delete_all/purge se quedaron bloqueados en el lock")
        self.assertEqual(result["purged"], 2)
        self.assertEqual(result["forced"], 1)
        self.assertEqual(manager.list(), [])

    def test_job_manager_status_exposes_worker_queue_and_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = JobManager(root / "uploads", root / "out")
            with patch.object(manager, "_process", side_effect=RuntimeError("boom")):
                job = manager.submit("libro.epub", b"dummy", JobOptions())
                for _ in range(200):
                    if job.status == "error":
                        break
                    import time

                    time.sleep(0.01)

            status = manager.status()

        self.assertEqual(job.status, "error")
        self.assertTrue(status["worker"]["alive"])
        self.assertEqual(status["worker"]["state"], "idle")
        self.assertIsNone(status["worker"]["current_job"])
        self.assertEqual(status["worker"]["processed"], 1)
        self.assertEqual(status["jobs"]["error"], 1)
        self.assertEqual(status["jobs"]["queued"], 0)
        self.assertIn("default", status["device"])
        self.assertIsInstance(status["device"]["model_loaded_on"], list)
        self.assertTrue(any("boom" in line for line in status["events"]))
        self.assertTrue(any("Servidor iniciado" in line for line in status["events"]))

    def test_loaded_model_devices_reports_real_device_after_fallback(self):
        import ebook_generator.tts_xtts as tts_module
        from ebook_generator.tts_xtts import loaded_model_devices

        class FakeParam:
            class device:
                type = "cpu"

        fake = Mock()
        fake.ebook_device_label = None
        fake.synthesizer.tts_model.parameters.return_value = iter([FakeParam()])
        tts_module._MODEL_CACHE.clear()
        try:
            self.assertEqual(loaded_model_devices(), [])
            tts_module._MODEL_CACHE["mps"] = fake
            tts_module._MODEL_CACHE["cpu"] = fake
            fake.synthesizer.tts_model.parameters.side_effect = lambda: iter([FakeParam()])
            self.assertEqual(loaded_model_devices(), ["cpu"])
        finally:
            tts_module._MODEL_CACHE.clear()

    def test_pause_blocks_the_worker_between_fragments_and_resume_releases_it(self):
        import threading
        import time

        from ebook_generator.web.jobs import Job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = JobManager(root / "uploads", root / "out")
            # Trabajo registrado sin pasar por la cola: así el hilo del worker real no compite por el estado.
            job = Job(id="pausable", filename="libro.epub", epub_path=root / "libro.epub", out_dir=root / "out", options=JobOptions())
            manager._jobs[job.id] = job
            job.status = "synthesizing"
            manager._worker_state = "busy"
            if True:
                self.assertTrue(manager.pause(job.id))
                self.assertTrue(job.pause_requested)

                worker = threading.Thread(target=manager._check_control, args=(job, 0.01), daemon=True)
                worker.start()
                for _ in range(200):
                    if job.paused:
                        break
                    time.sleep(0.01)
                self.assertTrue(job.paused)
                self.assertTrue(worker.is_alive())
                self.assertEqual(manager.status()["worker"]["state"], "paused")

                self.assertTrue(manager.resume(job.id))
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertFalse(job.paused)
                self.assertTrue(any("Reanudado" in line for line in job.log))

                # Pausar un trabajo terminado no está permitido; cancelar uno pausado sí lo libera.
                self.assertTrue(manager.pause(job.id))
                self.assertTrue(manager.cancel(job.id))
                self.assertFalse(job.pause_requested)
                job.status = "done"
                self.assertFalse(manager.pause(job.id))

    def test_kill_port_command_requests_a_sigterm_to_lsof_pid(self):
        with patch.object(cli_module.shutil, "which", return_value="/usr/bin/lsof"):
            with patch.object(cli_module.subprocess, "run", return_value=Mock(returncode=0, stdout="123\n", stderr="")) as run_mock:
                with patch.object(cli_module.os, "kill") as kill_mock:
                    cli_module.kill_port(8000, force=False)

        run_mock.assert_called_once()
        kill_mock.assert_called_once_with(123, signal.SIGTERM)

    def test_read_toc_falls_back_to_spine_when_ncx_has_empty_navmap(self):
        # EPUB comercial con navMap vacío; no se versiona (derechos de autor), así que la prueba se omite sin él.
        path = Path("epub-sample/Nunca Me Abandones.epub")
        if not path.exists():
            self.skipTest(f"EPUB de prueba no disponible: {path}")
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())

            def read(name: str):
                for candidate in (name, name.replace("%20", " ")):
                    if candidate in names:
                        return zf.read(candidate)
                return None

            container = read("META-INF/container.xml")
            opf_path = etree.fromstring(container).xpath("string(//c:rootfile/@full-path)", namespaces=NS)
            opf_bytes = read(opf_path)
            opf = etree.fromstring(opf_bytes)

            manifest = {}
            for item in opf.xpath("//opf:manifest/opf:item", namespaces=NS):
                manifest[item.get("id", "")] = {
                    "href": item.get("href", ""),
                    "media_type": item.get("media-type", ""),
                    "properties": item.get("properties", ""),
                }

            def resolve(href: str, base: str = "") -> str:
                import posixpath
                joined = posixpath.join(base, href) if base else href
                return posixpath.normpath(joined)

            toc = _read_toc(opf, manifest, resolve, read)

        self.assertGreater(len(toc), 0)
        self.assertTrue(all(entry.title for entry in toc))


if __name__ == "__main__":
    unittest.main()
