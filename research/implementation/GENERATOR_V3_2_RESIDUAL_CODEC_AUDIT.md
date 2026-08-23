# Generator v3.2 residual/codec audit — AGRI_116

## Scope

This audit does not train, alter weights, decode generated WAVs, or create
v3.3. It checks the stored SPICOR token provenance and the current Qwen 12 Hz
codec path, then separates codec correctness from the observed residual-model
collapse.

## Stored-token provenance

The SPICOR debug arrays were produced by the Swara `Qwen12HzCodecAdapter`.
The recorded external asset is `Qwen/Qwen3-TTS-Tokenizer-12Hz`, revision
`7dd38ad4e9bad454aae9cd937d0cd577604fe229`, Apache-2.0. The local tokenizer
weight hash is:

```text
836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258
```

The local Qwen source checkout is at commit
`022e286b98fbec7e1e916cb940cdf532cd9f488e`; its tokenizer API and the stored
asset configuration agree on the 12 Hz tokenizer interface. The tokenizer
configuration reports 24 kHz input/output, 1,920-sample encode/decode stride,
12.5 Hz frame rate, 16 valid quantizers, and vocabulary size 2,048.

## AGRI_116 stored versus fresh encode

Prepared audio: `data/spicor_eng_m_spk001_v1/audio_24k/IISc_SPICORProject_EN_M_AGRI_116.wav`

* stored shape: `(53, 16)`, `int16`, IDs 9–2047
* fresh shape: `(53, 16)`, same orientation and dtype/range
* stored versus fresh: **exact equality for all 848 values**
* codebook-by-codebook equality: **16/16 codebooks, 100%**

The fresh token sequence decoded through the same current local adapter to a
24 kHz, 4.24-second waveform with RMS `0.132792`. The prepared source RMS was
`0.136486`; source/decode waveform correlation was `0.9289`, MSE `0.00259`.
This is a valid codec roundtrip and rules out a stored-array revision,
orientation, codebook-order, offset, or special-token incompatibility for
AGRI_116.

## Token geometry and distributions

The stored array is raw frame-major data: row `t` contains codebooks 0–15 for
frame `t`; there is no BOS/EOS row or codebook offset. AGRI_116 has 53 frames,
consistent with 4.24 seconds at 12.5 Hz. Each codebook has roughly 46–53
unique IDs and approximately 5.43–5.73 bits of empirical entropy in this
utterance. The residual streams are therefore not intrinsically one-ID or
empty in the target data.

## Interpreting the listening result

Because the stored tokens exactly reproduce a fresh encode and decode with the
current tokenizer, the report that a separately produced `A_ground_truth_full`
file is unintelligible cannot be attributed to the stored SPICOR arrays or the
current codec revision. The most likely fault is in that artifact's decode
construction: a codebook/frame transpose, a different decoder/API revision,
or an output assembled from a different token buffer. The adapter must receive
rank-two `[frames, codebooks]` data with all 16 codebooks in their original
order.

The intelligible `C_generated_primary_GT_residual` result has a different
implication: codebook 0 carries the dominant semantic/phonetic trajectory, and
the target residual streams can restore acoustic detail when paired with a
valid primary stream. It does **not** show that target residuals are invalid,
nor that the target full-token decode is incompatible.

## Residual predictor findings

The user-observed free-running residual collapse (several codebooks with only
1–4 predicted IDs) is a generator failure, not a codec failure. Target residual
codebooks contain broad per-utterance variation, while the v3.2 generated
residual streams collapse. Exact teacher-forced and free-running per-codebook
accuracy could not be recomputed in this checkout because the trained v3.2
`best.pt` is not locally available; those measurements require the external
Drive checkpoint. The audit therefore does not invent per-codebook accuracies
or a precise first-collapse index.

## Conclusion

* Stored-token codec provenance: **verified**.
* Fresh encode/decode: **PASS**.
* Stored-versus-fresh match: **exact**.
* CB0: valid raw primary/semantic stream; it is the dominant intelligibility
  carrier.
* Residuals: valid target codec streams; generated residual predictor collapses.
* First collapsing generated codebook: **not determinable without v3.2
  checkpoint outputs** (user reports several codebooks at 1–4 IDs).

### Recommended single next intervention

Do not change the codec or reinterpret stored tokens. After retrieving the
v3.2 checkpoint, run per-codebook teacher-forced/free-running diagnostics. If
the collapse is confirmed, the single next intervention should target the
residual predictor's training signal/conditioning (starting with codebook-1
teacher-forced behavior), not the tokenizer or token layout.
