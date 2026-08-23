# Swara N1 versus NeuTTS: failure analysis

This is a source-based comparison of the local read-only NeuTTS repository at `/Users/saikiran/Documents/tts-reference/neutts` (Apache-2.0 code). No Swara code, weights, or training were changed.

## NeuTTS generation formulation (confirmed from source)

The central implementation is `neutts/neutts/neutts.py`:

1. `NeuTTS.infer()` calls `_apply_chat_template(ref_codes, ref_text, text, emotion)` and then `_infer_torch()`.
2. For the phoneme models (`neutts-nano` and `neutts-air`), `_apply_chat_template()` calls `_to_phones()` on both `ref_text` and target `input_text`, concatenates them with a space, and tokenizes the resulting phoneme string.
3. It constructs the chat prefix `user: Convert the text to speech:<|TEXT_PROMPT_START|> ... <|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>`.
4. It converts every reference NeuCodec ID to a dedicated textual token `<|speech_N|>` and appends those tokens immediately after `<|SPEECH_GENERATION_START|>`.
5. `_infer_torch()` calls the causal Hugging Face model's `generate()` with this whole prefix, `use_cache=True`, an EOS token `<|SPEECH_GENERATION_END|>`, and a minimum of 50 new tokens. The generated continuation is decoded and all `<|speech_N|>` IDs are passed to `codec.decode_code()`.

Concrete symbolic prompt for the phoneme path (abbreviated only for readability):

```text
user: Convert the text to speech:
<|TEXT_PROMPT_START|>
[phonemes(ref_text)] [phonemes(target_text)]
<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>
<|speech_1842|><|speech_77|>...<|speech_...|>
```

The model then predicts the continuation:

```text
<|speech_next|> ... <|SPEECH_GENERATION_END|>
```

The exact functions are `_to_phones`, `_apply_chat_template`, `_infer_torch`, and `_decode` in `neutts/neutts/neutts.py`. Reference codes are produced by `NeuTTS.encode_reference()`, which encodes 16-kHz mono audio with NeuCodec and returns a one-dimensional code sequence.

## Causal dependency and previous audio history

NeuTTS is a unified causal language-model stream. At every generated speech position, the causal prefix contains:

```text
all prompt text/control tokens
all reference speech codec tokens
all previously generated speech codec tokens
```

Thus the operational dependency is:

```text
P(speech_t | phoneme(ref_text), phoneme(target_text), controls,
                  reference_speech_0:t_ref,
                  generated_speech_0:t-1)
```

The reference speech prefix is not merely a speaker-ID lookup: source confirms that its actual NeuCodec IDs are inserted into the LM context. Speaker/style/prosody anchoring and codec-distribution anchoring are therefore plausible consequences of the prefix; the source does not separately label or disentangle those factors.

Swara N1 instead computed a frame-position-indexed classifier:

```text
P(codec_t | linguistic_sequence, fixed_frame_position_t)
```

It had no prior speech tokens, no reference prefix, and no causal speech history. Its causal Transformer mask operated over frame positions, but each position was independently derived from text/position rather than from an acoustic token stream.

## Training sequence and shift (confirmed)

`examples/finetune.py::preprocess_sample()` constructs one ordinary causal-LM sequence:

```text
user: Convert the text to speech:<|TEXT_PROMPT_START|>{phonemes}
<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>
{<|speech_i|> for every target codec ID}
<|SPEECH_GENERATION_END|>
```

The code pads to `max_seq_len` and sets `labels` to `-100` everywhere before the first `<|SPEECH_GENERATION_START|>` position. Labels from the speech-generation-start position onward are copied from `input_ids`; padding is masked by `attention_mask`. Hugging Face causal-LM loss performs the standard one-position shift internally, so the input token at position `t` predicts the token at `t+1`.

Confirmed consequences:

- True previous speech tokens are teacher-forced during training because they are earlier tokens in the same `input_ids` stream.
- Text/control tokens and the prompt prefix receive no loss.
- Speech loss begins at the speech-generation start region (the start token itself is included in labels by this preprocessing; the model's internal shift determines which prediction consumes it).
- EOS is present in the training target when the dataset code includes it and is the explicit inference stop token in `_infer_torch()`.

The exact dataset packing beyond this example script is not a published independent trainer contract; the statements above are directly confirmed by `preprocess_sample()`.

## Reference speech role and zero-reference behavior

Source-confirmed: normal `NeuTTS.infer()` requires `ref_codes` and `ref_text`; the public examples encode or load a reference before generation. The prompt always includes the reference code IDs for this path. The source does not document a zero-reference or learned-speaker-ID mode for the Nano/Air causal model. A zero-length reference might be technically constructible, but is not an approved/verified inference mode and should be treated as unsupported. The reference transcript is concatenated with target phonemes, so the LM also sees the linguistic content corresponding to the reference prefix.

## Unified vocabulary and special tokens

`TRAINING.md` explicitly describes adding 65,536 tokens `<|speech_0|>` through `<|speech_65535|>` plus structural tokens such as `<|TEXT_PROMPT_START|>`, `<|TEXT_PROMPT_END|>`, `<|SPEECH_GENERATION_START|>`, `<|SPEECH_GENERATION_END|>`, and `<|SPEECH_REPLACE|>`. The finetuning script uses `AutoTokenizer.encode()` on the complete mixed string and a single `AutoModelForCausalLM`/LM head. Therefore text/control/speech are represented in one expanded vocabulary and one next-token objective, rather than separate frame heads.

The source does not publish a separate numeric “speech offset”; speech IDs are tokenizer vocabulary entries and are serialized as `<|speech_N|>` strings. Exact assigned integers are tokenizer-checkpoint data, not hard-coded by the repository.

## Position and alignment

NeuTTS does **not** implement a fixed 50-Hz text-to-frame schedule, duration predictor, or phoneme-to-frame alignment table. Text is placed before the speech prefix in one causal sequence. Alignment is implicit: the LM learns when to emit speech tokens based on the complete phoneme/control context and its causal acoustic history. The 2048 context limit is documented in `README.md`; it covers prompt plus generated speech (about 30 seconds including prompt duration).

## Model scale (source-confirmed README)

`README.md` reports active and embedding-inclusive parameter counts:

| model | active parameters | embeddings + active | input |
|---|---:|---:|---|
| NeuTTS-Air | ~360M | ~552M | phonemes |
| NeuTTS-Nano | ~120M | ~229M | phonemes |
| NeuTTS-2E | ~125M | ~236M | text |

The local source does not include the Hugging Face Nano/Air `config.json`, so exact hidden width/layer/head counts and exact LM-head allocation cannot be confirmed from this checkout alone. The ~109M Nano gap between embedding-inclusive and active counts includes token embeddings/output vocabulary costs and other non-active parameters; it is already orders of magnitude above Swara N1's 1.38M backbone and N1-A's 8.45M flat head.

## Why NeuTTS remains on a speech manifold

Ranked architectural explanations, without proposing a fix:

1. **Causal previous codec-token history (highest confidence).** The next token is trained and generated in the same mixed stream, so every output is conditioned on a valid speech prefix.
2. **Reference codec prefix (high confidence for speaker/style anchoring).** The actual reference code sequence supplies a real codec-manifold continuation context and avoids starting from an unanchored arbitrary acoustic state.
3. **Unified next-token training and capacity (high confidence).** A single ~120M active causal LM is trained on the same sequence construction used at inference, with a 65,536-way speech vocabulary. N1 had a tiny backbone and frame-wise fixed conditioning, so its validation outputs can use legal IDs without learning legal transitions.
4. **Implicit alignment.** The model learns the text-to-speech timing through sequence continuation instead of being forced onto a hand-designed fixed frame schedule.

Training scale is also important empirically, but it is not an architectural mechanism and cannot be isolated from this source checkout.

## Direct comparison

| mechanism | Swara N1 | NeuTTS Nano/Air |
|---|---|---|
| text conditioning | Swara typed tokens projected to each fixed frame | phoneme prompt in one causal stream |
| previous audio history | none | all prior reference/generated speech tokens |
| reference audio | none | required prompt codec prefix in public path |
| alignment | fixed `floor(frame * text_len / frame_count)` | implicit causal continuation |
| training objective | frame-position CE (flat or FSQ heads) | ordinary next-token CE on speech suffix |
| speech target shift | no mixed text/audio shift | standard causal shift over unified sequence |
| speaker conditioning | none in N1 | reference code prefix; no separate speaker ID in source path |
| model scale | 1.38M backbone; 9.84M flat total | ~120M active Nano, ~360M Air |
| speech vocabulary | 65,536 IDs or 8×4 coordinates | 65,536 tokenizer speech entries |
| generation | text-only frame rollout | cached causal LM continuation with EOS |

## Top 3 structural differences most likely to explain N1 failure

1. N1 removed the entire previous-speech-token causal state; NeuTTS trains exactly on and generates from a valid codec prefix.
2. N1 used a fixed text-to-frame schedule; NeuTTS lets causal sequence modeling learn alignment and stopping.
3. N1's tiny model/head was trained as a frame classifier on a small corpus; NeuTTS uses a much larger unified speech-token LM and a reference prefix, giving substantially stronger transition/manifold capacity.

## Cheapest single-mechanism experiment

Keep Distill-NeuCodec, the Swara M1 linguistic sequence, the same 5-minute SPICOR split, same tokenizer-free text representation, and the same approximate tiny backbone. Add only a causal speech-token input stream: construct training rows as `[BOS, text/control conditioning, true codec_0, true codec_1, ...]` (or a text prefix followed by BOS and speech IDs), apply ordinary next-token CE only to speech targets, and teacher-force true prior codec IDs during training. At inference, begin with BOS plus text and feed each generated codec ID back as the next input; do not use reference audio, duration prediction, speaker modules, or a larger model. Pass requires: two-utterance free-running memorization, a materially lower held-out teacher-forced CE than N1, non-silent oracle-compatible decode, and validation generated bigram overlap/trajectory diversity substantially above N1 without a single shared trajectory. Fail if free-running outputs remain off-manifold or held-out teacher-forced accuracy remains effectively zero. This isolates “previous acoustic-token history” while holding codec, frontend, data, and scale fixed.

## Decision boundary

The evidence supports testing acoustic history first, but does not prove that history alone is sufficient: NeuTTS also benefits from a reference prefix, a much larger LM, and much larger training data. No Swara architecture was modified in this analysis.
