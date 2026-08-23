# Distill-NeuCodec acceptance record

## Model provenance

- Model: `neuphonic/distill-neucodec`
- Resolved revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`
- License: Apache-2.0
- Checkpoint size: 1,025,488,162 bytes
- Checkpoint SHA256: `adace21f679b30f071c02e0cb3502d965ab08b50be936a5e81944674a5ae101e`
- Runtime parameter count: 247,322,282

## Verified codec geometry

The accepted runtime uses 16 kHz encoder input and 24 kHz decoder output,
approximately 50 Hz frames, one codebook, and 65,536 token IDs. Runtime codes
have shape `[1, 1, T]`. The FSQ has eight scalar dimensions with four levels
each, so `4^8 = 65,536`; a token is structurally representable as
`[d0,d1,d2,d3,d4,d5,d6,d7]`, each `di ∈ {0,1,2,3}`.

## Roundtrip result

The fixed SPICOR panel achieved 20/20 encodes and 20/20 decodes. Mean CPU
encode RTF was approximately 0.213 and decode RTF approximately 0.086. No
corrupt or silent outputs were detected.

## Human blind listening

- Randomization seed: `20260823`
- Original preferred: 9/20
- Distill-NeuCodec preferred: 2/20
- No meaningful difference: 9/20

Observed codec-associated details: subtle stumbles in samples 05, 07 and 09;
slight distortion in 13; less clear pronunciation in 14; a longer pause in 15;
and clearer original pronunciation in 17. Positive observations included codec
preference for 02 and 03, and clearer “internet” in the codec reconstruction of
10. There was no systematic intelligibility, pronunciation, speaker-identity,
or obvious-corruption failure.

**Listening gate: PASS.** The codec is not transparent/perfect, but is
sufficiently faithful for the next bounded Swara experiment.

## Architectural rationale

The previous Qwen representation was 12.5 Hz × 16 codebooks × 2,048 IDs and
required primary plus CB1–CB15 residual generation. Swara's compact residual
path collapsed from CB1; independent residual heads did not solve validation
learning. Distill-NeuCodec changes this to a 50 Hz single stream with no
residual-codebook chain.

## N1 hypothesis (not a claim)

N1 compares one flat 65,536-way classifier against eight independent 4-way FSQ
coordinate classifiers using the same backbone, data, schedule, optimizer,
seed, and future training budget. The eight dimensions are not assumed to be
statistically independent; this is only an output-representation hypothesis.

## Decision

Codec research status: **FROZEN FOR N1**. Selected codec:
`neuphonic/distill-neucodec`, because provenance, runtime, roundtrip, and human
listening gates passed. No further codec hunting is authorized during N1
unless new evidence invalidates this decision.

## Linked artifacts

- `experiments/codec_bakeoff_v1/manifest.json`
- `experiments/codec_bakeoff_v1/reports/distill_neucodec_metrics.json`
- `experiments/codec_bakeoff_v1/reports/distill_neucodec_blind_mapping.json`
- `experiments/codec_bakeoff_v1/reports/DISTILL_NEUCODEC_ROUNDTRIP_REPORT.md`
- `experiments/codec_bakeoff_v1/DISTILL_NEUCODEC_LISTENING_SCORECARD.md`
