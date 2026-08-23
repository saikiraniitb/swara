# Alignment Gate A manual review

For each numbered JSON file:

1. Open the `source_wav` path from the JSON and listen while following
   `authoritative_transcript`.
2. Inspect every grapheme unit's `start_seconds` and `end_seconds`.
3. Confirm word boundaries, punctuation-owned gaps, leading silence, and
   trailing silence.
4. Record any word whose cut begins late, ends early, includes an adjacent word,
   or is assigned to the wrong spoken content.
5. Review the four contextual WAVs in `low_confidence_cuts/` first. They are
   diagnostics only and do not replace full-utterance review.

Machine validity is 10/10, but this directory has not been manually approved.
The next action is review—not 30-minute preprocessing or model work.
