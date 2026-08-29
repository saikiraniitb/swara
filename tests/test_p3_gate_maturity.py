import math
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_swara_speech_poc_p3_colab import archive_protocol_bug_step_one, evaluation_stop_reason, paths  # noqa: E402


def evaluation_payload(*, similarity=1.0, valid=True, ce=11.0):
    validation = {
        "total_loss": ce + 1.0,
        "duration": {"smooth_l1": 1.0},
        "acoustic": {"ce": ce},
    }
    free = {
        "max_nonself_similarity": similarity,
        "minimum_text_swap_change": 0.9,
        "pathological_loop": False,
        "maximum_shared_prefix_frames": 0,
        "repetition": {
            "generated_self_transition_rate": 0.1,
            "longest_generated_run": 2,
        },
        "rows": [{
            "ground_truth_duration": {"valid_ids": valid},
            "full_pipeline": {"valid_ids": valid},
        }],
    }
    return validation, free


class P3GateMaturityTests(unittest.TestCase):
    def test_step_one_similarity_is_diagnostic_only(self):
        validation, free = evaluation_payload(similarity=1.0)
        self.assertIsNone(evaluation_stop_reason(1, validation, free))

    def test_step_250_similarity_activates_quality_stop(self):
        validation, free = evaluation_payload(similarity=1.0)
        self.assertEqual(evaluation_stop_reason(250, validation, free), "max_nonself_similarity")

    def test_nonfinite_metric_stops_at_step_one(self):
        validation, free = evaluation_payload(ce=math.nan)
        self.assertEqual(evaluation_stop_reason(1, validation, free), "non_finite_evaluation_metric")

    def test_invalid_generated_ids_stop_at_step_one(self):
        validation, free = evaluation_payload(valid=False)
        self.assertEqual(evaluation_stop_reason(1, validation, free), "invalid_generated_ids")

    def test_known_step_one_protocol_bug_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            drive = paths(Path(directory))
            drive["run_state"].write_text('{"current_optimizer_step": 1, "stop_reason": "max_nonself_similarity"}')
            drive["recovery"].write_bytes(b"old recovery")
            archived = archive_protocol_bug_step_one(drive)
            self.assertIsNotNone(archived)
            self.assertTrue((Path(archived) / "run_state/p3_run_state.json").is_file())
            self.assertTrue((Path(archived) / "run_state/recovery_latest.pt").is_file())
            self.assertTrue(drive["state_dir"].is_dir())
            self.assertFalse(drive["run_state"].exists())

    def test_unknown_nonzero_run_is_never_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            drive = paths(Path(directory))
            drive["run_state"].write_text('{"current_optimizer_step": 250, "stop_reason": null}')
            with self.assertRaisesRegex(RuntimeError, "not the known step-1"):
                archive_protocol_bug_step_one(drive)


if __name__ == "__main__":
    unittest.main()
