# Pronunciation Contract v0

## Decision: F. Hybrid representation

Swara v0 uses **canonical grapheme tokens plus optional explicit pronunciation-token spans**, all scoped by language spans and anchored to the original text. A pronunciation token alphabet is an owned, finite, versioned internal symbol set; it is not raw IPA and not an opaque learned-token vocabulary.

Canonical text stays available for semantics, punctuation, display, and fallback. Explicit pronunciation replaces only the pronunciation realization of an affected span. The generator consumes the compiled linguistic sequence, not free-form text or raw IPA.

## Strategy comparison

| Strategy | Assessment for Swara | Decision |
|---|---|---|
| A. Pure graphemes | Small/simple but leaves Indian names and irregular English implicit; no deterministic override | Reject |
| B. Pure phonemes | Deterministic but loses original spelling/semantics and requires complete G2P/inventories before v0 | Reject |
| C. IPA | Human-readable to specialists but large/normalization-sensitive and inconvenient as direct model vocabulary | Reject as internal model format |
| D. Learned pronunciation tokens | Compact after training but opaque, difficult to manually override, and data/training dependent | Reject |
| E. Graphemes + pronunciation override | Solves named exceptions but underspecifies language spans and normal linguistic structure | Insufficient alone |
| F. Hybrid representation | Preserves text, permits deterministic local realization, scales by language, and keeps v0 small | **Select** |

The chosen design has a modest tokenizer increase only for a finite set of control/language/pronunciation symbols. It avoids forcing a complete pan-Indian phoneme inventory into v0, supports a Transformer through ordinary token sequences, adds negligible inference work relative to speech generation, and allows users/systems to override a span directly.

## Contract types

```yaml
PronunciationDocument:
  schema_version: swara.pronunciation.v0
  source_text: string                 # immutable user-visible input
  normalized_text: string             # canonical rendering after text normalization
  sentences: [Sentence]
  spans: [LanguageSpan]
  overrides: [PronunciationOverride]
  linguistic_tokens: [LinguisticToken] # compiled output consumed by generator adapter

Sentence:
  id: string
  source_range: {start: int, end: int} # Unicode code-point offsets in source_text
  terminal_punctuation: string | null

LanguageSpan:
  id: string
  source_range: {start: int, end: int}
  language: bcp47                     # e.g. en-IN, hi, te; never infer silently after compile
  script: string                       # e.g. Latn, Deva, Telu

PronunciationOverride:
  id: string
  source_range: {start: int, end: int}
  pronunciation_system: string         # e.g. swara-phones-v0
  tokens: [string]                     # finite, validated internal symbols
  language: bcp47
  source: user | lexicon | system
  priority: integer

LinguisticToken:
  kind: grapheme | pronunciation | punctuation | boundary | control
  value: string
  language: bcp47 | null
  source_range: {start: int, end: int} | null
  override_id: string | null
```

`source_range` makes every rendered token traceable to original input; normalizer expansion (for example a number) uses generated tokens with the originating source range.

## Coordinate system and normalization mapping

Public text ranges are defined against `source_text` using Python Unicode string/code-point offsets (`[start, end)`). Normalized coordinates are internal and derived through `NormalizationMap`; they are never the primary external API.

The M1 normalizer returns `NormalizedDocument(source_text, normalized_text, source_map)`. Its map records the source range that produced each normalized character and deterministically projects spans in either direction. A source span must fully contain every normalized-character origin it touches. If a conservative normalization collapse/composition makes a requested source span partial, empty, or ambiguous, projection raises a typed error rather than shifting the override.

## Compilation rules

1. Preserve `source_text` unchanged.
2. Normalize text into a canonical spoken form while retaining source ranges.
3. Segment sentences, punctuation, and explicit/recognized language spans.
4. Validate source-coordinate overrides, project them through `NormalizationMap`, then validate their language and token alphabet against the resulting normalized spans.
5. Emit pronunciation tokens only for resolved override/lexicon/G2P spans; emit canonical grapheme tokens elsewhere in v0.
6. Emit punctuation and sentence-boundary tokens explicitly. These are linguistic timing cues, not styling prompts.
7. Fail closed on malformed overlaps or unsupported pronunciation-system versions; do not silently drop a user override.

## Example

Input: `Saikiran travelled from Hyderabad.`

```yaml
source_text: "Saikiran travelled from Hyderabad."
spans:
  - {id: l1, source_range: {start: 0, end: 34}, language: en-IN, script: Latn}
overrides:
  - id: p1
    source_range: {start: 0, end: 8}
    pronunciation_system: swara-phones-v0
    tokens: [SAI, KI, RAN]             # illustrative symbols, not an inventory decision
    language: en-IN
    source: user
    priority: 100
linguistic_tokens:
  - {kind: pronunciation, value: SAI, language: en-IN, source_range: {start: 0, end: 8}, override_id: p1}
  - {kind: pronunciation, value: KI, language: en-IN, source_range: {start: 0, end: 8}, override_id: p1}
  - {kind: pronunciation, value: RAN, language: en-IN, source_range: {start: 0, end: 8}, override_id: p1}
  - {kind: grapheme, value: travelled, language: en-IN, source_range: {start: 9, end: 18}, override_id: null}
  - {kind: grapheme, value: from, language: en-IN, source_range: {start: 19, end: 23}, override_id: null}
  - {kind: grapheme, value: Hyderabad, language: en-IN, source_range: {start: 24, end: 33}, override_id: null}
  - {kind: punctuation, value: ".", language: null, source_range: {start: 33, end: 34}, override_id: null}
  - {kind: boundary, value: sentence_end, language: null, source_range: null, override_id: null}
```

The example deliberately does not assert a pronunciation for Hyderabad. A future lexicon/G2P can replace the grapheme span while preserving the same contract.

## V0 commitments and deferrals

V0_REQUIRED: English/Indian-English span handling, manual override representation and validation, punctuation/sentence boundaries, and a versioned token mapping.

Deferred: automatic Indian-name lexicon, number/acronym verbalization breadth, Hindi/English and Telugu/English G2P, language identification confidence policy, and all exhaustive phoneme inventories. The document schema already accommodates them.
