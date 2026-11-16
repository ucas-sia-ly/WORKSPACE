"""CPU-only portable deployment checks; no torch, CUDA or network required."""
from contextlib import asynccontextmanager
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[3] / "adapters/ccr_remote_qwen.py"
SPEC = importlib.util.spec_from_file_location("ccr_remote_qwen_tested", ENTRY)
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def safetensor(path, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({key: dict(dtype="BF16", shape=[1], data_offsets=[0, 2])}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


class PortableQwenTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ccr-portable-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.prepared = self.root / "blocks"
        self.lora = self.root / "lora" / remote.IDENTITY["lora_filename"]
        self.manifest = self.root / "bundle.json"
        safetensor(self.model / "text_encoder/model.safetensors", "encoder.weight")
        safetensor(self.model / "vae/model.safetensors", "vae.weight")
        for name in ("tokenizer/tokenizer_config.json", "processor/processor_config.json"):
            write_json(self.model / name, {})
        write_json(self.model / "transformer/config.json", {"num_layers": 2})
        safetensor(self.lora, "img_in.lora_up.weight")
        records = {}
        for name, key in (("block_0.safetensors", "transformer_blocks.0.weight"),
                          ("block_1.safetensors", "transformer_blocks.1.weight"),
                          ("non_block.safetensors", "img_in.weight")):
            path = self.prepared / name
            safetensor(path, key)
            header, digest = remote.read_header(path)
            records[name] = dict(size=path.stat().st_size, sha256=remote.file_sha256(path),
                                 header_sha256=digest, tensor_count=len(header))
        _, lora_header = remote.read_header(self.lora)
        self.original = dict(complete=True, files=records, identity=dict(
            format_version=1, dtype="BF16", lora_merged=True, lora_strength=1.0,
            num_layers=2, config_sha256=remote.file_sha256(self.model / "transformer/config.json"),
            sources=[dict(path="/original/source/shard.safetensors", size=1000,
                          mtime_ns=123, header_sha256="a" * 64)],
            lora=dict(path="/original/lightning.safetensors", size=self.lora.stat().st_size,
                      mtime_ns=456, header_sha256=lora_header)))
        write_json(self.prepared / remote.MANIFEST, self.original)
        self.refresh_bundle()

    def refresh_bundle(self, updates=None):
        files = {}
        for directory in (self.model, self.prepared, self.lora.parent):
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    files[path.relative_to(self.root).as_posix()] = dict(
                        size=path.stat().st_size, sha256=remote.file_sha256(path))
        self.bundle = dict(format_version=1, kind=remote.KIND, **remote.IDENTITY,
                           paths=dict(model="model", prepared="blocks", lora=self.lora.relative_to(self.root).as_posix()),
                           prepared_manifest_sha256=remote.file_sha256(self.prepared / remote.MANIFEST), files=files)
        if updates:
            self.bundle.update(updates)
        self.seal_bundle()

    def seal_bundle(self):
        self.bundle.pop("result_sha256", None)
        self.bundle["result_sha256"] = remote.fingerprint(self.bundle)
        write_json(self.manifest, self.bundle)
        self.expected = remote.file_sha256(self.manifest)

    def validate(self):
        return remote.validate_bundle(self.manifest, self.expected, self.model, self.lora, self.prepared)

    def test_full_validation_without_original_transformer_payload(self):
        original, result = self.validate()
        self.assertEqual(original, self.original)
        self.assertFalse(Path(original["identity"]["sources"][0]["path"]).exists())
        self.assertFalse(list((self.model / "transformer").glob("*.safetensors")))
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["verified_files"], len(self.bundle["files"]))
        self.assertEqual(result["source_payload_policy"], "original_transformer_payload_not_deployed")
        self.assertTrue(result["original_source_identity_preserved"])
        self.assertTrue(result["lora_already_merged"])

    def test_manifest_external_anchor_and_internal_seal(self):
        self.expected = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            self.validate()
        self.bundle["result_sha256"] = "f" * 64
        write_json(self.manifest, self.bundle)
        self.expected = remote.file_sha256(self.manifest)
        with self.assertRaisesRegex(ValueError, "seal"):
            self.validate()

    def test_same_size_payload_corruption_detected_on_every_start(self):
        self.validate()
        path = self.prepared / "block_1.safetensors"
        value = path.read_bytes()
        path.write_bytes(value[:-1] + b"\1")
        with self.assertRaisesRegex(ValueError, "SHA256/size"):
            self.validate()

    def test_no_unlisted_config_or_weights_and_no_original_dit_shards(self):
        path = self.model / "extra.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            self.validate()
        path.unlink()
        safetensor(self.model / "transformer/original.safetensors", "base.weight")
        self.refresh_bundle()
        with self.assertRaisesRegex(ValueError, "original DiT payload"):
            self.validate()

    def test_unsafe_paths_and_symlinks_rejected(self):
        for unsafe in ("../escape", "/etc/passwd", "model/../escape", "model//extra", "model\\extra"):
            with self.subTest(path=unsafe):
                self.bundle["files"][unsafe] = dict(size=0, sha256="0" * 64)
                self.seal_bundle()
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.validate()
                del self.bundle["files"][unsafe]
        self.seal_bundle()
        (self.model / "alias").symlink_to(self.root / "blocks", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.validate()

    def test_runtime_paths_and_exact_model_identity(self):
        with self.assertRaisesRegex(ValueError, "Runtime weight paths"):
            remote.validate_bundle(self.manifest, self.expected, self.root, self.lora, self.prepared)
        self.refresh_bundle(dict(model_revision="different"))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate()

    def test_explicit_root_with_nested_manifest_and_safe_relative_paths(self):
        nested = self.root / "outputs/deployment/weight_bundle.json"
        nested.parent.mkdir(parents=True)
        self.manifest.replace(nested)
        self.manifest = nested
        with patch.dict(os.environ, {"CCR_BUNDLE_ROOT": str(self.root)}):
            self.assertEqual(self.validate()[1]["status"], "verified")
        with self.assertRaisesRegex(ValueError, "absolute directory"):
            remote.validate_bundle(self.manifest, self.expected, self.model, self.lora, self.prepared,
                                   bundle_root="relative")
        with self.assertRaisesRegex(ValueError, "outside"):
            remote.validate_bundle(self.manifest, self.expected, self.model, self.lora, self.prepared,
                                   bundle_root=self.model)

    def test_prepared_record_cross_check_and_unmerged_rejection(self):
        self.original["files"]["block_0.safetensors"]["header_sha256"] = "b" * 64
        write_json(self.prepared / remote.MANIFEST, self.original)
        self.refresh_bundle()
        with self.assertRaisesRegex(ValueError, "payload/header/provenance"):
            self.validate()
        self.original["identity"]["lora_merged"] = False
        write_json(self.prepared / remote.MANIFEST, self.original)
        self.refresh_bundle()
        with self.assertRaisesRegex(ValueError, "BF16/LoRA provenance"):
            self.validate()

    def test_missing_component_and_wrong_block_count(self):
        self.bundle["paths"]["prepared"] = "missing"
        self.seal_bundle()
        with self.assertRaisesRegex(ValueError, "Runtime weight paths"):
            self.validate()
        self.refresh_bundle()
        self.original["files"].pop("block_1.safetensors")
        write_json(self.prepared / remote.MANIFEST, self.original)
        self.refresh_bundle()
        with self.assertRaisesRegex(ValueError, "block inventory"):
            self.validate()

    def environment(self):
        return dict(CCR_BUNDLE_MANIFEST=str(self.manifest), CCR_BUNDLE_SHA256=self.expected,
                    LIGHTX2V_DISK_OFFLOAD="1", LIGHTX2V_MEMORY_PROFILE="disk24g")

    def test_adapter_hook_returns_original_manifest_and_reports_failed_validation(self):
        with patch.dict(os.environ, self.environment()):
            self.assertEqual(remote.validate_prepared_weights(self.model, self.lora, self.prepared), self.original)
            self.assertEqual(remote._VALIDATION["status"], "verified")
            self.manifest.write_text("{}")
            with self.assertRaises(ValueError):
                remote.validate_prepared_weights(self.model, self.lora, self.prepared)
            self.assertEqual(remote._VALIDATION["status"], "failed")

    def test_explicit_profile_and_original_generate_handler_preserved(self):
        @asynccontextmanager
        async def lifespan(app):
            yield

        def generate(request: dict):
            return request

        fake = SimpleNamespace(MODEL_ID=remote.IDENTITY["model_id"], MODEL_REVISION=remote.IDENTITY["model_revision"],
            LORA_ID=remote.IDENTITY["lora_id"], LORA_REVISION=remote.IDENTITY["lora_revision"],
            LORA_FILENAME=remote.IDENTITY["lora_filename"], LIGHTX2V_COMMIT=remote.IDENTITY["lightx2v_commit"],
            INFER_STEPS=4, GUIDANCE_SCALE=1.0, CANVAS_POLICY="source_aspect_v1",
            lifespan=lifespan, generate=generate, health=lambda: dict(status="ok", sampling=dict(infer_steps=4)))
        with patch.dict(os.environ, self.environment()):
            app = remote.create_app(fake)
            self.assertIs(fake.validate_prepared_weights, remote.validate_prepared_weights)
            self.assertIs(next(route.endpoint for route in app.routes if route.path == "/generate"), generate)
            health = next(route.endpoint for route in app.routes if route.path == "/health")()
            self.assertEqual(health["sampling"], dict(infer_steps=4))
            self.assertEqual(health["deployment"], dict(kind=remote.KIND, manifest_sha256=self.expected,
                                                       weight_dtype="BF16", validation="full-file-sha256"))
            remote._VALIDATION = dict(status="verified", elapsed_seconds=1.2)
            second = next(route.endpoint for route in app.routes if route.path == "/health")()
            self.assertEqual(second["deployment"], health["deployment"])
            diagnostics = next(route.endpoint for route in app.routes if route.path == "/deployment-validation")()
            self.assertEqual(diagnostics["validation"]["status"], "verified")
            os.environ["LIGHTX2V_MEMORY_PROFILE"] = "resident_bf16_48g"
            with self.assertRaisesRegex(ValueError, "explicit disk24g"):
                remote.create_app(fake)

    def test_main_is_loopback_only(self):
        uvicorn = SimpleNamespace(run=unittest.mock.Mock())
        with patch.dict("sys.modules", uvicorn=uvicorn), patch.object(remote, "create_app", return_value="app"), \
                patch.dict(os.environ, {"LIGHTX2V_PORT": "9123"}):
            remote.main()
        uvicorn.run.assert_called_once_with("app", host="127.0.0.1", port=9123)

    def test_cpu64_profile_is_explicit_and_reports_its_own_stable_performance(self):
        from generation import qwen_cpu_cache
        @asynccontextmanager
        async def lifespan(app):
            yield

        base_loader = unittest.mock.Mock(return_value="pipe")
        fake = SimpleNamespace(MODEL_ID=remote.IDENTITY["model_id"], MODEL_REVISION=remote.IDENTITY["model_revision"],
            LORA_ID=remote.IDENTITY["lora_id"], LORA_REVISION=remote.IDENTITY["lora_revision"],
            LORA_FILENAME=remote.IDENTITY["lora_filename"], LIGHTX2V_COMMIT=remote.IDENTITY["lightx2v_commit"],
            INFER_STEPS=4, GUIDANCE_SCALE=1.0, CANVAS_POLICY="source_aspect_v1", load_pipeline=base_loader,
            lifespan=lifespan, generate=lambda request: request, health=lambda: dict(status="ok"))
        environment = {**self.environment(), "LIGHTX2V_MEMORY_PROFILE": "disk24g_cpu64",
                       "LIGHTX2V_DISK_MODEL_PATH": str(self.prepared)}
        with patch.dict(os.environ, environment), patch.object(qwen_cpu_cache, "install_cpu_phase_cache") as install:
            app = remote.create_app(fake)
            remote._VALIDATION = dict(status="failed")
            with self.assertRaisesRegex(RuntimeError, "successful portable bundle"):
                fake.load_pipeline()
            install.assert_not_called()
            remote._VALIDATION = dict(status="verified")
            self.assertEqual(fake.load_pipeline(), "pipe")
            install.assert_called_once_with("pipe", str(self.prepared), cache_phases=True)
            health = next(route.endpoint for route in app.routes if route.path == "/health")()
            self.assertEqual(health["performance"], qwen_cpu_cache.performance_identity())
            self.assertEqual(health["performance"]["cpu_dit_budget_bytes"], 24 * 2**30)
            self.assertNotIn("memory_profile", health["deployment"])
            os.environ["LIGHTX2V_MEMORY_PROFILE"] = "disk24g"
            baseline = remote.create_app(fake)
            self.assertIs(fake.load_pipeline, base_loader)
            baseline_health = next(route.endpoint for route in baseline.routes if route.path == "/health")()
            self.assertNotIn("performance", baseline_health)

    def test_dit_only_cache_profile_preserves_original_phase_loading(self):
        from generation import qwen_cpu_cache
        @asynccontextmanager
        async def lifespan(app):
            yield

        fake = SimpleNamespace(MODEL_ID=remote.IDENTITY["model_id"], MODEL_REVISION=remote.IDENTITY["model_revision"],
            LORA_ID=remote.IDENTITY["lora_id"], LORA_REVISION=remote.IDENTITY["lora_revision"],
            LORA_FILENAME=remote.IDENTITY["lora_filename"], LIGHTX2V_COMMIT=remote.IDENTITY["lightx2v_commit"],
            INFER_STEPS=4, GUIDANCE_SCALE=1.0, CANVAS_POLICY="source_aspect_v1",
            load_pipeline=unittest.mock.Mock(return_value="pipe"), lifespan=lifespan,
            generate=lambda request: request, health=lambda: dict(status="ok"))
        environment = {**self.environment(), "LIGHTX2V_MEMORY_PROFILE": "disk24g_ditcache24",
                       "LIGHTX2V_DISK_MODEL_PATH": str(self.prepared)}
        with patch.dict(os.environ, environment), patch.object(qwen_cpu_cache, "install_cpu_phase_cache") as install:
            app = remote.create_app(fake)
            remote._VALIDATION = dict(status="verified")
            fake.load_pipeline()
            install.assert_called_once_with("pipe", str(self.prepared), cache_phases=False)
            identity = next(route.endpoint for route in app.routes if route.path == "/health")()["performance"]
            self.assertEqual(identity, qwen_cpu_cache.performance_identity("disk24g_ditcache24"))
            self.assertFalse(identity["cache_encoder"])
            self.assertFalse(identity["cache_vae"])


if __name__ == "__main__":
    unittest.main()
