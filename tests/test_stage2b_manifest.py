import copy
import json
from pathlib import Path
import unittest

import soundfile as sf

from swara.training.stage2b_manifest import (
    Stage2BManifestError,
    validate_accepted_manifest,
    validate_candidate_manifest,
    validate_candidate_record,
    validate_disjoint_splits,
    validate_transfer_texts,
)
from swara.frontend.spans import TextSpan
from swara.training.stage2b_pronunciation import TrainingPronunciationTarget


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/stage2b_pronunciation/candidate_manifest.jsonl"
ACCEPTED = ROOT / "data/stage2b_pronunciation/accepted_manifest.jsonl"
MECHANISM = ROOT / "data/stage2b_pronunciation/stage2b4b_manifest.json"
FIXTURES = ROOT / "data/stage2b_pronunciation/evaluation_fixtures.json"
CLIP_MANIFEST = ROOT / "data/stage2b_pronunciation/review_clips/clip_manifest.jsonl"
DECISIONS = ROOT / "data/stage2b_pronunciation/human_decisions.jsonl"
PHONE_REVIEW = ROOT / "data/stage2b_pronunciation/lexical_phone_review.json"
UNSUPPORTED_VARIANT = "UNSUPPORTED_ALPHABET_VARIANT"


def load_candidates():
    with MANIFEST.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Stage2BManifestTests(unittest.TestCase):
    def test_candidate_manifest_is_valid_and_audio_backed(self):
        records = load_candidates()
        validate_candidate_manifest(records, repository_root=ROOT)
        self.assertEqual(len(records), 20)
        self.assertEqual(sum(r["verification_status"] == "VERIFIED" for r in records), 16)
        self.assertEqual(sum(r["verification_status"] == UNSUPPORTED_VARIANT for r in records), 1)
        self.assertEqual(sum(r["verification_status"] == "REJECTED" for r in records), 3)
        for record in records:
            self.assertEqual(record["target_text"], record["transcript"][record["source_span_start"]:record["source_span_end"]])
            self.assertTrue((ROOT / record["audio_path"]).is_file())
            self.assertLess(record["audio_start_seconds"], record["audio_end_seconds"])
            self.assertLess(record["codec_frame_start"], record["codec_frame_end"])

    def test_accepted_manifest_contains_only_verified_targets_and_excludes_c002(self):
        accepted = load_jsonl(ACCEPTED)
        validate_accepted_manifest(accepted, repository_root=ROOT)
        self.assertEqual(len(accepted), 16)
        self.assertNotIn("s2b4b-cand-002", {record["candidate_id"] for record in accepted})
        for record in accepted:
            target = TrainingPronunciationTarget(
                source_span=TextSpan(record["source_span_start"], record["source_span_end"], record["target_text"]),
                override_id=record["override_id"],
                verified_phone_sequence=tuple(record["verified_phone_sequence"]),
                audio_start_seconds=record["audio_start_seconds"],
                audio_end_seconds=record["audio_end_seconds"],
                codec_frame_start=record["codec_frame_start"],
                codec_frame_end=record["codec_frame_end"],
                alignment_confidence=record["alignment_confidence"],
                alignment_source=record["alignment_source"],
                alignment_version=record["alignment_version"],
                codec_frame_rate_hz=record["codec_frame_rate_hz"],
                codec_total_frames=record["codec_total_frames"],
            )
            self.assertEqual(target.source_span.start, record["source_span_start"])
        unsupported = copy.deepcopy(load_candidates()[1])
        with self.assertRaises(Stage2BManifestError):
            validate_accepted_manifest([unsupported], repository_root=ROOT)

    def test_unsupported_phone_symbol_is_rejected(self):
        record = copy.deepcopy(load_candidates()[0])
        record["candidate_id"] = "invalid-phone"
        record["proposed_phone_sequence"] = ["NOT_IN_SWARA_PHONES_V0"]
        with self.assertRaises(Stage2BManifestError):
            validate_candidate_record(record, repository_root=ROOT)

    def test_verified_record_requires_phone_sequence_and_override(self):
        record = copy.deepcopy(load_candidates()[0])
        record["candidate_id"] = "unverified-as-verified"
        record["verification_status"] = "VERIFIED"
        record["override_id"] = None
        record["proposed_phone_sequence"] = None
        with self.assertRaises(Stage2BManifestError):
            validate_candidate_record(record, repository_root=ROOT)

    def test_duplicate_candidate_ids_are_rejected(self):
        records = load_candidates()[:2]
        records[1]["candidate_id"] = records[0]["candidate_id"]
        with self.assertRaises(Stage2BManifestError):
            validate_candidate_manifest(records, repository_root=ROOT)

    def test_train_and_eval_occurrence_overlap_is_rejected(self):
        record = load_candidates()[0]
        with self.assertRaises(Stage2BManifestError):
            validate_disjoint_splits([record], [record])

    def test_transfer_text_must_not_duplicate_training_transcript(self):
        with self.assertRaises(Stage2BManifestError):
            validate_transfer_texts(["training sentence"], ["training sentence"])
        validate_transfer_texts(["training sentence"], ["new transfer sentence"])

    def test_every_pending_candidate_has_one_bounded_review_clip(self):
        candidates = {r["candidate_id"]: r for r in load_candidates() if r["verification_status"] != "REJECTED"}
        clips = load_jsonl(CLIP_MANIFEST)
        self.assertEqual({r["candidate_id"] for r in clips}, set(candidates))
        for clip in clips:
            self.assertTrue((ROOT / clip["review_clip_path"]).is_file())
            info = sf.info(ROOT / clip["review_clip_path"])
            self.assertGreater(info.frames, 0)
            self.assertLessEqual(clip["review_clip_start_seconds"], clip["aligned_start_seconds"])
            self.assertGreaterEqual(clip["review_clip_end_seconds"], clip["aligned_end_seconds"])
            self.assertEqual(clip["source_span"], [candidates[clip["candidate_id"]]["source_span_start"], candidates[clip["candidate_id"]]["source_span_end"]])
        rejected_ids = {r["candidate_id"] for r in load_candidates() if r["verification_status"] == "REJECTED"}
        self.assertTrue(rejected_ids.isdisjoint({r["candidate_id"] for r in clips}))

    def test_review_templates_match_frozen_candidates(self):
        candidates = {r["candidate_id"]: r for r in load_candidates() if r["verification_status"] != "REJECTED"}
        decisions = load_jsonl(DECISIONS)
        self.assertEqual({r["candidate_id"] for r in decisions}, set(candidates))
        self.assertEqual(sum(r["decision"] == "VERIFIED" for r in decisions), 16)
        self.assertEqual(sum(r["decision"] == UNSUPPORTED_VARIANT for r in decisions), 1)
        variants = [variant for target in json.loads(PHONE_REVIEW.read_text(encoding="utf-8"))["targets"] for variant in target["variants"]]
        self.assertEqual(len(variants), 11)
        self.assertEqual(sum(variant["status"] == "VERIFIED" for variant in variants), 10)
        self.assertEqual(sum(variant["status"] == UNSUPPORTED_VARIANT for variant in variants), 1)

    def test_human_audio_decisions_are_complete_and_phone_verified(self):
        records = load_candidates()
        decisions = load_jsonl(DECISIONS)
        self.assertEqual(len(decisions), 17)
        self.assertEqual({r["candidate_id"] for r in decisions}, {r["candidate_id"] for r in records if r["verification_status"] != "REJECTED"})
        self.assertTrue(all(r["spoken_target_correct"] is True for r in decisions))
        self.assertTrue(all(r["alignment_accepted"] is True for r in decisions))
        self.assertTrue(all(r["pronunciation_clear"] is True for r in decisions))
        self.assertEqual(sum(r["decision"] == "VERIFIED" for r in decisions), 16)
        self.assertEqual(sum(r["decision"] == UNSUPPORTED_VARIANT for r in decisions), 1)

        relations = {record["review_label"]: record["pronunciation_relation"] for record in decisions}
        self.assertEqual(relations["C001"], "distinct_from:s2b4b-cand-002")
        self.assertEqual(relations["C002"], "distinct_from:s2b4b-cand-001")
        self.assertEqual(relations["C003"], "distinct_from:s2b4b-cand-004")
        self.assertEqual(relations["C004"], "distinct_from:s2b4b-cand-003")
        for first, second in (("C005", "C006"), ("C007", "C008"), ("C009", "C010"), ("C011", "C012"), ("C013", "C014"), ("C015", "C016")):
            self.assertEqual(relations[first], f"same_as:s2b4b-cand-{int(second[1:]):03d}")

    def test_lexical_variants_preserve_same_and_different_pronunciations(self):
        payload = json.loads(PHONE_REVIEW.read_text(encoding="utf-8"))
        variants = [variant for target in payload["targets"] for variant in target["variants"]]
        self.assertEqual(len(variants), 11)
        self.assertEqual({variant["variant_id"] for variant in variants}, {
            "Agrawal-A", "Agrawal-B", "Singh-A", "Singh-B", "Kumar-A",
            "Sharma-A", "Gupta-A", "Mumbai-A", "Kashmir-A", "Mishra-A", "Sensharma-A",
        })
        self.assertEqual(next(v for v in variants if v["variant_id"] == "Agrawal-A")["candidate_labels"], ["C001"])
        self.assertEqual(next(v for v in variants if v["variant_id"] == "Agrawal-B")["candidate_labels"], ["C002"])
        self.assertEqual(next(v for v in variants if v["variant_id"] == "Singh-A")["candidate_labels"], ["C003"])
        self.assertEqual(next(v for v in variants if v["variant_id"] == "Singh-B")["candidate_labels"], ["C004"])
        self.assertEqual(next(v for v in variants if v["variant_id"] == "Kumar-A")["candidate_labels"], ["C005", "C006"])
        self.assertEqual(sum(v["status"] == "VERIFIED" for v in variants), 10)
        self.assertEqual(sum(v["status"] == UNSUPPORTED_VARIANT for v in variants), 1)
        self.assertEqual(sum(v["verified_phone_sequence"] is None for v in variants), 1)
        labels = {label for variant in variants for label in variant["candidate_labels"]}
        self.assertEqual(labels, {f"C{i:03d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20)})

    def test_frozen_mechanism_split_is_disjoint_and_excludes_unsupported_variant(self):
        payload = json.loads(MECHANISM.read_text(encoding="utf-8"))
        train = set(payload["train_candidate_ids"])
        evaluation = set(payload["eval_seen_candidate_ids"])
        accepted = set(payload["accepted_candidate_ids"])
        self.assertEqual(len(accepted), 16)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(evaluation), 6)
        self.assertEqual(train & evaluation, set())
        self.assertEqual(train | evaluation, accepted)
        self.assertNotIn("s2b4b-cand-002", accepted | train | evaluation)
        self.assertEqual(len(payload["trained_variant_ids"]), 10)
        self.assertEqual({item["candidate_id"] for item in payload["accepted_occurrences"]}, accepted)
        self.assertTrue(all(item["audio_path"] and item["codec_frame_end"] > item["codec_frame_start"] for item in payload["accepted_occurrences"]))
        accepted_by_id = {record["candidate_id"]: record for record in load_jsonl(ACCEPTED)}
        validate_disjoint_splits(
            [accepted_by_id[candidate_id] for candidate_id in train],
            [accepted_by_id[candidate_id] for candidate_id in evaluation],
        )
        fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
        trained_targets = {payload["variants"][variant_id]["target_text"] for variant_id in payload["trained_variant_ids"]}
        self.assertTrue(all(len(fixtures["transfer"].get(target, ())) >= 2 for target in trained_targets))
        training_transcripts = [record["transcript"] for record in accepted_by_id.values() if record["candidate_id"] in train]
        transfer_texts = [text for texts in fixtures["transfer"].values() for text in texts]
        validate_transfer_texts(training_transcripts, transfer_texts)


if __name__ == "__main__":
    unittest.main()
