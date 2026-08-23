# Alignment Gate B manual review

Machine alignment is complete; this panel is preliminary until reviewed.

The sole review question is:

> Does the alignment correctly correspond to the spoken words and silence?

## Panel construction

The fixed seed is `20260823`. Thirty unique rows were selected in six ordered,
deduplicated buckets, five rows each:

1. lowest-confidence rows;
2. Indian-name/location-heavy rows;
3. fastest lexical speaking-rate rows;
4. slowest lexical speaking-rate rows;
5. punctuation/silence-heavy rows; and
6. ordinary random controls.

`review_manifest.json` records the exact selection and source WAV references.
Each numbered `.md` file is a readable unit table; the matching `.json` retains
the complete character-level and unit-level contract.

## Review procedure

1. Open the referenced full source WAV; source audio was not copied or changed.
2. Follow the authoritative transcript and check each word's start/end seconds.
3. Check leading/trailing silence and punctuation-owned gaps.
4. Pay special attention to rows with confidence or duration flags.
5. Use `suspicious_cuts/` only as a convenience. Each cut includes 100 ms of
   context on both sides and does not replace full-utterance review.
6. Record misplaced, truncated, merged, or wrong-word spans. Low confidence by
   itself is not a failure if the timestamp is correct.

Do not assess TTS quality, pronunciation quality, or speaker preference here.
Do not begin model implementation until this panel is accepted.

