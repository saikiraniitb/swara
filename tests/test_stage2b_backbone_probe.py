import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "stage2b_backbone_probe.py"
SPEC = importlib.util.spec_from_file_location("stage2b_backbone_probe", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage2BBackboneProbeTests(unittest.TestCase):
    def test_probe_is_local_metadata_only_and_reports_qwen_dimensions(self):
        result = MODULE.probe_qwen_local(ROOT / "models" / "qwen3-tts-12hz-0.6b-base")
        self.assertTrue(result["local_asset_complete"], result["missing_files"])
        self.assertEqual(result["config"]["talker_hidden_size"], 1024)
        self.assertEqual(result["config"]["talker_text_hidden_size"], 2048)
        self.assertEqual(result["config"]["talker_num_code_groups"], 16)
        self.assertIsInstance(result["model_tensor_parameter_count"], int)
        self.assertIn("no model weights loaded", result["probe_scope"])

    def test_probe_does_not_select_a_backend_for_stage2b(self):
        source = PROBE_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("qwenfoundationtts", source)
        self.assertNotIn("speechgenerator", source)
