# V0 Success Criteria

V0 is successful only when all criteria below are met in one controlled architectural smoke test. Intelligibility is the priority; production naturalness, Indian pronunciation coverage, clone fidelity, speed, and streaming are explicitly not acceptance criteria.

| Criterion | Binary evidence |
|---|---|
| Swara-owned end-to-end pipeline | One `SpeechRequest` enters the Swara orchestrator and produces a waveform artifact plus `RenderReport`. |
| Frontend boundary works | The report contains `NormalizedText`, `PronunciationDocument`, and `LinguisticSequence` versions/hashes produced by Swara components. |
| Explicit override is represented | A request contains a span-anchored pronunciation override; compiled linguistic tokens contain the corresponding validated pronunciation tokens and provenance. |
| Speaker interface exists | The request uses `speaker_id`; the report records resolution to a `SpeakerCondition` without exposing an upstream model-specific public API. |
| Generator path works | Generator returns audio-token frames that pass `AudioTokenSpec` validation: correct rank, codebook count, vocabulary bounds, and end condition. |
| Codec path works | Codec decodes those validated frames into a nonempty PCM waveform with declared sample rate. |
| No Dia runtime dependency | Dependency/import audit shows no `dia`, `dac`, or Dia checkpoint/runtime call on the v0 render path. |
| No Qwen runtime dependency in public API | Public request, control, frontend, generator, codec, and report types contain no Qwen class/model ID/tokenizer type. An internal prototype adapter is permitted only if it conforms to these owned interfaces. |
| Failure behavior works | Unsupported language, malformed override, unknown speaker, invalid token frame, and missing component each fail before producing misleading output and return a typed error. |

## Explicit non-criteria

No acceptance threshold is set yet for WER, MOS, Indian-name accuracy, latency, model size, voice similarity, expressive range, long-form continuity, or mobile deployment. Those require later evaluation design and are not evidence that the architecture contract works.

