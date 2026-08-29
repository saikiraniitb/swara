# Stage2B.4B pronunciation-target human review

Status: `HUMAN_PHONE_REVIEW_COMPLETE`

This is the completed human-review package and its linked mechanism manifest.
The candidate records were prepared from local IISc/SPICOR transcript and
audio inventory only. No phone sequence was inferred from spelling, ASR,
alignment, an LLM, or the audio model; the sequences below were supplied by the
human reviewer. `accepted_manifest.jsonl` contains only the 16 VERIFIED
occurrences.

## Preparation and filtering

- Transcript inventory: `data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl`.
- Audio used for candidates: `data/spicor_eng_m_spk001_v1/audio_24k/`.
- The inventory contains many transcript-only records; only records with a
  local prepared WAV were eligible.
- Lexical intervals were prepared with
  `src/swara/alignment/ctc_forced.py`, using the local
  `facebook/wav2vec2-base-960h` checkpoint at revision
  `22aad52d435eb6dbaf354bdad9b0da84ce7d615c`.
- Qwen codec geometry was measured from the local tokenizer asset. The target
  rate is 12.5 Hz, with 16 codebooks; frame ranges use the frozen
  `floor(start * 12.5)` / `ceil(end * 12.5)` rule.
- The local aligner emitted a warning that `masked_spec_embed` was newly
  initialized. This makes human listening review mandatory; it is not evidence
  of pronunciation correctness.
- The automatic preselection threshold is confidence `>= 0.60`. This is a
  conservative candidate filter chosen after inspecting the observed
  distribution, not a verification criterion.
- Review-only clips are now available under
  `data/stage2b_pronunciation/review_clips/`. Each contains the aligned
  interval with approximately 100--200 ms context and does not replace or
  modify canonical source audio.

## Requested lexical panel occurrence audit

Counts below distinguish full-corpus transcript occurrences from records with
an actually available local prepared audio file.

| target | full-corpus train source IDs | local audio-backed source IDs |
|---|---:|---:|
| Kolkata | 18 | 0 |
| Bengaluru | 25 | 0 |
| Prayagraj | 1 | 0 |
| Ajinkya | 1 | 0 |
| Banerjee | 16 | 0 |
| Anirban | 1 | 0 |
| Arundhati | 4 | 0 |
| Ashutosh | 6 | 0 |
| Konkona | 6 | 1 |
| Sensharma | 2 | 1 |

The requested panel therefore does not provide the desired audio-backed
Kolkata/Bengaluru/Ajinkya examples in this checkout. The one local occurrence
of `Konkona` and `Sensharma` is in the same sentence and is retained only as a
review candidate; it is not trusted pronunciation supervision.

## Candidate review table

Human phone transcription is complete for the 10 accepted pronunciation
variants. Agrawal B remains a distinct human-confirmed pronunciation but is
`UNSUPPORTED_ALPHABET_VARIANT` and is excluded from training.

| candidate | target | source audio ID | source span | aligned seconds | confidence | Qwen frames | status |
|---|---|---|---|---|---:|---|---|
| s2b4b-cand-001 | Agrawal | AGRI_107 | [95,102) | 5.304452--5.684771 | 0.7733 | [66,72) | VERIFIED |
| s2b4b-cand-002 | Agrawal | ENTE_6231 | [119,126) | 7.274742--7.655514 | 0.6627 | [90,96) | UNSUPPORTED_ALPHABET_VARIANT |
| s2b4b-cand-003 | Singh | AGRI_1732 | [4,9) | 0.481093--0.681548 | 0.6002 | [6,9) | VERIFIED |
| s2b4b-cand-004 | Singh | AGRI_4510 | [77,82) | 5.567836--5.828202 | 0.7709 | [69,73) | VERIFIED |
| s2b4b-cand-005 | Kumar | ENTE_7184 | [26,31) | 1.781301--2.061505 | 0.9048 | [22,26) | VERIFIED |
| s2b4b-cand-006 | Kumar | FOOD_973 | [88,93) | 5.012923--5.293646 | 0.9643 | [62,67) | VERIFIED |
| s2b4b-cand-007 | Sharma | AGRI_423 | [14,20) | 1.042767--1.363619 | 0.6987 | [13,18) | VERIFIED |
| s2b4b-cand-008 | Sharma | FOOD_7155 | [32,38) | 2.103643--2.404164 | 0.8578 | [26,31) | VERIFIED |
| s2b4b-cand-009 | Gupta | ENTE_6063 | [67,72) | 3.446137--3.766708 | 0.9527 | [43,48) | VERIFIED |
| s2b4b-cand-010 | Gupta | FOOD_4112 | [43,48) | 2.525112--2.805680 | 0.9054 | [31,36) | VERIFIED |
| s2b4b-cand-011 | Mumbai | AGRI_5618 | [38,44) | 2.405534--2.686180 | 0.8910 | [30,34) | VERIFIED |
| s2b4b-cand-012 | Mumbai | ENTE_315 | [73,79) | 4.408417--4.789144 | 0.6511 | [55,60) | VERIFIED |
| s2b4b-cand-013 | Kashmir | ENTE_137 | [128,135) | 6.817626--7.218663 | 0.8154 | [85,91) | VERIFIED |
| s2b4b-cand-014 | Kashmir | ENTE_138 | [133,140) | 6.568794--6.949303 | 0.9189 | [82,87) | VERIFIED |
| s2b4b-cand-015 | Mishra | AGRI_6732 | [49,55) | 3.186824--3.447382 | 0.7925 | [39,44) | VERIFIED |
| s2b4b-cand-016 | Mishra | FOOD_2846 | [25,31) | 2.087121--2.327943 | 0.6318 | [26,30) | VERIFIED |
| s2b4b-cand-017 | Kapoor | AGRI_6792 | [8,14) | 0.622205--0.883130 | 0.2082 | [7,12) | REJECTED |
| s2b4b-cand-018 | Kapoor | ENTE_4277 | [53,59) | 3.771395--4.072304 | 0.5222 | [47,51) | REJECTED |
| s2b4b-cand-019 | Konkona | WEAT_3645 | [56,63) | 3.671228--4.072455 | 0.5000 | [45,51) | REJECTED |
| s2b4b-cand-020 | Sensharma | WEAT_3645 | [64,73) | 4.172761--4.654234 | 0.6235 | [52,59) | VERIFIED |

The three rejected rows failed the automatic confidence prefilter. They have
not been marked phonetically unsupported; they are simply not in the first
review panel.

## Completed human decision state

All 17 human-reviewed occurrences have clear spoken targets and accepted
alignment. Sixteen have valid human-supplied `swara-phones-v0` sequences and
are in `accepted_manifest.jsonl`. C002 / Agrawal B is retained as a distinct
human-confirmed pronunciation but is excluded as
`UNSUPPORTED_ALPHABET_VARIANT`.

No selected candidate is a possessive-root case such as `Kolkata's`; therefore
the first panel does not silently label a possessive suffix as part of a root
pronunciation. Any future morphological candidate must refine the root span,
label the suffix explicitly, or be rejected.

## Current counts

- Automatic candidates: 20.
- Human audio/alignment accepted: 17.
- Verified pronunciation occurrences: 16.
- Unsupported alphabet variant: 1 (C002 / Agrawal B).
- Pre-existing trusted verified items: 0; all verified labels are supplied by
  the current human review provenance.
- Rejected automatically: 3.
- Currently eligible and present in `accepted_manifest.jsonl`: 16.
- Explicit pronunciation variants: 11.
- Distinct reviewed lexical targets: 9 (Agrawal, Singh, Kumar, Sharma, Gupta,
  Mumbai, Kashmir, Mishra, Sensharma).
- Phone-sequence representability: 10 variants representable; Agrawal B is
  unsupported as a duration distinction. No alphabet change is authorized.

## Frozen mechanism split

The accepted split is frozen in `stage2b4b_manifest.json`: 10 TRAIN
occurrences and 6 EVAL-SEEN occurrences. The split has no `(source_id, source
span)` overlap. The frozen text fixtures are in `evaluation_fixtures.json` and
are not copied from training transcripts.

## `swara-phones-v0` inventory reference

This is the actual inventory in `src/swara/frontend/pronunciation.py`, also
documented in `research/pronunciation/ALPHABET_V0.md`. The repository gives
group-level descriptions only; it does not provide target-specific
pronunciations for this review set.

| symbols | documented meaning/example |
|---|---|
| `A AA E EE I II O OO U UU AI AU` | vowel-like units; no per-symbol target example is documented |
| `K G T D N P B M Y R L V S H SH CH J NG` | common consonant-like units; no per-symbol target example is documented |

The repository's example `(S, AI, K, I, R, A, N)` for “Saikiran” is explicitly
an architecture/testing example and is not evidence for any candidate here.
The inventory omits stress, tone, aspiration detail, schwa policy, and many
English/Indian phonetic distinctions. No inventory modification is permitted
for this review.

## Candidate-by-candidate listening sheet

The following 17 candidates are the frozen pending set. Each clip contains the
aligned interval plus up to 150 ms context on either side. `clip_manifest.jsonl`
contains the complete machine-readable provenance. The canonical source WAVs
remain the training-data references; review clips are not training audio. The
audio-review decisions are recorded in `human_decisions.jsonl`; the phone
sequence remains intentionally blank.

### C001 — Agrawal

Target: `Agrawal`  
Transcript: “The occasion was marked with the presence of Agriculture and Water Resource Minister Brijmohan Agrawal, Parliamentary Secretary Tokhan Sahu, M I A Shrichand Sundrani and others”  
Review clip: `data/stage2b_pronunciation/review_clips/C001_Agrawal.wav`  
Alignment confidence: `0.7733`  
Aligned interval: `5.304452–5.684771 s`  
Codec frames: `[66,72)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C002 — Agrawal

Target: `Agrawal`  
Transcript: “In response to the discussion held at legislative Assembly on Friday, Minister for the aforesaid departments Brijmohan Agrawal talked at first about the”  
Review clip: `data/stage2b_pronunciation/review_clips/C002_Agrawal.wav`  
Alignment confidence: `0.6627`  
Aligned interval: `7.274742–7.655514 s`  
Codec frames: `[90,96)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C003 — Singh

Target: `Singh`  
Transcript: “Lal Singh flags off first batch of Amaranth Ji Yatra from Lakhanpur”  
Review clip: `data/stage2b_pronunciation/review_clips/C003_Singh.wav`  
Alignment confidence: `0.6002`  
Aligned interval: `0.481093–0.681548 s`  
Codec frames: `[6,9)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C004 — Singh

Target: `Singh`  
Transcript: “Add pureed fresh fruits like strawberry and papaya to curd, suggests Jasmine Singh of Ozone”  
Review clip: `data/stage2b_pronunciation/review_clips/C004_Singh.wav`  
Alignment confidence: `0.7709`  
Aligned interval: `5.567836–5.828202 s`  
Codec frames: `[69,73)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C005 — Kumar

Target: `Kumar`  
Transcript: “D S P law and order Bimal Kumar who is investigating the Bagbera incident said that so far no arrest has been made”  
Review clip: `data/stage2b_pronunciation/review_clips/C005_Kumar.wav`  
Alignment confidence: `0.9048`  
Aligned interval: `1.781301–2.061505 s`  
Codec frames: `[22,26)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C006 — Kumar

Target: `Kumar`  
Transcript: “At the current fund management fee, all the fund managers are bleeding, said Shailendra Kumar, managing director, S B I Pension Funds Private limited”  
Review clip: `data/stage2b_pronunciation/review_clips/C006_Kumar.wav`  
Alignment confidence: `0.9643`  
Aligned interval: `5.012923–5.293646 s`  
Codec frames: `[62,67)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C007 — Sharma

Target: `Sharma`  
Transcript: “TRAI chairman Sharma, who has accused Apple of colonising data, may be looking at regulating internet giants on this pretext”  
Review clip: `data/stage2b_pronunciation/review_clips/C007_Sharma.wav`  
Alignment confidence: `0.6987`  
Aligned interval: `1.042767–1.363619 s`  
Codec frames: `[13,18)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C008 — Sharma

Target: `Sharma`  
Transcript: “Details of the notes seized and Sharma were provided to the Income Tax department, said Pathak”  
Review clip: `data/stage2b_pronunciation/review_clips/C008_Sharma.wav`  
Alignment confidence: `0.8578`  
Aligned interval: `2.103643–2.404164 s`  
Codec frames: `[26,31)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C009 — Gupta

Target: `Gupta`  
Transcript: “When contacted the Principal of the college Doctor Pradeep Bharati Gupta and Medical Superintendent ”  
Review clip: `data/stage2b_pronunciation/review_clips/C009_Gupta.wav`  
Alignment confidence: `0.9527`  
Aligned interval: `3.446137–3.766708 s`  
Codec frames: `[43,48)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C010 — Gupta

Target: `Gupta`  
Transcript: “Principal Commissioner Income Tax Sangeeta Gupta was the chief guest on the occasion”  
Review clip: `data/stage2b_pronunciation/review_clips/C010_Gupta.wav`  
Alignment confidence: `0.9054`  
Aligned interval: `2.525112–2.805680 s`  
Codec frames: `[31,36)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C011 — Mumbai

Target: `Mumbai`  
Transcript: “At the Mint Luxury Conference held in Mumbai last week, I moderated a conversation with Milner on why manners maketh a man”  
Review clip: `data/stage2b_pronunciation/review_clips/C011_Mumbai.wav`  
Alignment confidence: `0.8910`  
Aligned interval: `2.405534–2.686180 s`  
Codec frames: `[30,34)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C012 — Mumbai

Target: `Mumbai`  
Transcript: “I train with Striders, a fitness and marathon training club, in Powai in Mumbai”  
Review clip: `data/stage2b_pronunciation/review_clips/C012_Mumbai.wav`  
Alignment confidence: `0.6511`  
Aligned interval: `4.408417–4.789144 s`  
Codec frames: `[55,60)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C013 — Kashmir

Target: `Kashmir`  
Transcript: “A law meant to aid the recovery of debts but could result in the backdoor acquisition of property by non Kashmiris in Jammu and Kashmir”  
Review clip: `data/stage2b_pronunciation/review_clips/C013_Kashmir.wav`  
Alignment confidence: `0.8154`  
Aligned interval: `6.817626–7.218663 s`  
Codec frames: `[85,91)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C014 — Kashmir

Target: `Kashmir`  
Transcript: “On the Udhampur Srinagar Baramulla rail link project, he said that the work was halted because of a law and order issue in Jammu and Kashmir”  
Review clip: `data/stage2b_pronunciation/review_clips/C014_Kashmir.wav`  
Alignment confidence: `0.9189`  
Aligned interval: `6.568794–6.949303 s`  
Codec frames: `[82,87)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C015 — Mishra

Target: `Mishra`  
Transcript: “The B J P leader said that Union minister Kalraj Mishra will go to Himachal Pradesh while Water Resources minister Uma Bharai will visit Assam”  
Review clip: `data/stage2b_pronunciation/review_clips/C015_Mishra.wav`  
Alignment confidence: `0.7925`  
Aligned interval: `3.186824–3.447382 s`  
Codec frames: `[39,44)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C016 — Mishra

Target: `Mishra`  
Transcript: “Unlikely, says Neelkanth Mishra of Credit Suisse, because of the low demand for credit now”  
Review clip: `data/stage2b_pronunciation/review_clips/C016_Mishra.wav`  
Alignment confidence: `0.6318`  
Aligned interval: `2.087121–2.327943 s`  
Codec frames: `[26,30)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

### C020 — Sensharma

Target: `Sensharma`  
Transcript: “Lipstick Under My Burkha box office collection day six- Konkona Sensharma film had a strong Wednesday”  
Review clip: `data/stage2b_pronunciation/review_clips/C020_Sensharma.wav`  
Alignment confidence: `0.6235`  
Aligned interval: `4.172761–4.654234 s`  
Codec frames: `[52,59)`  
Current phone sequence: see final PHONE TRANSCRIPTION REVIEW below

Human decisions:

- [x] SPOKEN_TARGET_CORRECT
- [ ] ALIGNMENT_BAD
- [x] PRONUNCIATION_CLEAR
- [ ] PRONUNCIATION_AMBIGUOUS
- [ ] REJECT

Human pronunciation note:

---

## Lexical comparison summary

Human review found two distinct variants for Agrawal and two distinct variants
for Singh. They remain separate. The other reviewed occurrences are grouped
only where the human decision explicitly said the pronunciation was the same.

- **Agrawal A:** C001 — `A G R A V AA L`
- **Agrawal B:** C002 — unsupported alphabet variant; no sequence
- **Singh A:** C003 — `S I NG`
- **Singh B:** C004 — `S I NG H`
- **Kumar A:** C005, C006 — `K UU M AA R`
- **Sharma A:** C007, C008 — `SH A R M AA`
- **Gupta A:** C009, C010 — `G UU P T AA`
- **Mumbai A:** C011, C012 — `M A M B AI`
- **Kashmir A:** C013, C014 — `K A SH M EE R`
- **Mishra A:** C015, C016 — `M I SH R A`
- **Sensharma A:** C020 — `S E N SH A R M AA`

No repository artifact was found that contains an explicitly human-verified
phone sequence for any of these exact lexical targets. The existing test
overrides are plumbing fixtures only and are not reused as lexical labels.

## PHONE TRANSCRIPTION REVIEW

Exactly 11 human-reviewed pronunciation variants are recorded below. The ten
representable variants have human-supplied phone sequences. Agrawal B remains
explicitly separate with no phone sequence because the current alphabet cannot
safely encode the confirmed distinction.

| variant_id | target_text | candidate_ids | representative_clip | verified_phone_sequence | status |
|---|---|---|---|---|---|
| Agrawal-A | Agrawal | C001 | `data/stage2b_pronunciation/review_clips/C001_Agrawal.wav` | `A G R A V AA L` | VERIFIED |
| Agrawal-B | Agrawal | C002 | `data/stage2b_pronunciation/review_clips/C002_Agrawal.wav` | `null` | UNSUPPORTED_ALPHABET_VARIANT |
| Singh-A | Singh | C003 | `data/stage2b_pronunciation/review_clips/C003_Singh.wav` | `S I NG` | VERIFIED |
| Singh-B | Singh | C004 | `data/stage2b_pronunciation/review_clips/C004_Singh.wav` | `S I NG H` | VERIFIED |
| Kumar-A | Kumar | C005/C006 | `data/stage2b_pronunciation/review_clips/C005_Kumar.wav` | `K UU M AA R` | VERIFIED |
| Sharma-A | Sharma | C007/C008 | `data/stage2b_pronunciation/review_clips/C007_Sharma.wav` | `SH A R M AA` | VERIFIED |
| Gupta-A | Gupta | C009/C010 | `data/stage2b_pronunciation/review_clips/C009_Gupta.wav` | `G UU P T AA` | VERIFIED |
| Mumbai-A | Mumbai | C011/C012 | `data/stage2b_pronunciation/review_clips/C011_Mumbai.wav` | `M A M B AI` | VERIFIED |
| Kashmir-A | Kashmir | C013/C014 | `data/stage2b_pronunciation/review_clips/C013_Kashmir.wav` | `K A SH M EE R` | VERIFIED |
| Mishra-A | Mishra | C015/C016 | `data/stage2b_pronunciation/review_clips/C015_Mishra.wav` | `M I SH R A` | VERIFIED |
| Sensharma-A | Sensharma | C020 | `data/stage2b_pronunciation/review_clips/C020_Sensharma.wav` | `S E N SH A R M AA` | VERIFIED |
