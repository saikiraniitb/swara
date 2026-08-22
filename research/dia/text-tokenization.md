# Dia Text Handling / Tokenization

## Summary answer (upfront)

**Dia uses raw UTF-8 byte-level "tokenization" — there is no subword
tokenizer, no phoneme/G2P step, and no pronunciation dictionary anywhere in
this repository.** The entire "tokenizer" is 12 lines of code:
`Dia._encode_text`, `dia/model.py:240-263`.

```python
def _encode_text(self, text: str) -> torch.Tensor:
    max_len = self.config.encoder_config.max_position_embeddings
    byte_text = text.encode("utf-8")
    replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
    text_tokens = list(replaced_bytes)
    return torch.tensor(text_tokens[:max_len], dtype=torch.long, device=self.device)
```

## Tokenizer implementation

- **Vocabulary**: every possible byte value, `0-255` — confirmed by
  `EncoderConfig.vocab_size: int = 256` (`dia/config.py:54`) feeding directly
  into `Encoder.embedding = nn.Embedding(enc_config.vocab_size, ...)`
  (`dia/layers.py:600-604`). This is a byte-level embedding table, one row
  per possible byte, no merges/BPE table, no vocabulary file anywhere in the
  repo.
- **Encoding granularity**: character/subword/phoneme is moot here — it's
  literally **bytes** of the UTF-8 encoding of the string. A single non-ASCII
  Unicode character (e.g. an accented letter, a Devanagari character) becomes
  **multiple** byte tokens (2-4 bytes per UTF-8 encoding rules), each
  consuming one embedding-table lookup and one position in the 1024-token
  encoder context window.
- **Unicode handling**: `str.encode("utf-8")` is the only Unicode-aware step.
  No normalization (no NFC/NFKC), no case-folding, no special handling of
  combining characters, no transliteration. Whatever bytes the Python
  `str.encode("utf-8")` call produces are what the model sees.
- **Unknown-token behavior**: there is no "unknown token" concept — because
  the vocabulary is the full byte range, *any* UTF-8-encodable string is
  representable without OOV. The failure mode instead is **train/test
  distribution mismatch**: if the training corpus never contained certain
  byte sequences (e.g. Devanagari script bytes, since the model "only
  supports English generation" per the README), the corresponding embedding
  rows are essentially untrained/random and the model's behavior on them is
  undefined-but-not-crashing (the architecture has no way to *refuse*, it
  will attempt to decode Model output regardless).
- **Punctuation**: passed through as ordinary bytes, no special treatment.
- **Truncation**: `text_tokens[:max_len]` — silently truncates any text
  whose UTF-8 byte length exceeds 1024 tokens, with **no warning or error**
  in `_encode_text` itself.
- **Padding**: separate step, `Dia._pad_text_input` (`dia/model.py:265-280`),
  pads with byte value `0` (the NUL byte) up to `max_len=1024`. Padding
  status is tracked via `padding_mask = (cond_src.squeeze(1) != 0)`
  (`dia/state.py:60`) — i.e. **byte value 0 is reserved as the pad sentinel**
  and is assumed never to appear in real input text (true in practice, since
  NUL bytes essentially never occur in natural-language text).

## Special control tokens

| Marker | Handling |
|---|---|
| `[S1]` | Replaced (as a literal 4-character ASCII substring) with byte `0x01` before UTF-8 byte-listing. Consumes exactly **one** embedding-table slot, not four. `dia/model.py:257` |
| `[S2]` | Replaced with byte `0x02`, same treatment. |
| Non-verbal tags: `(laughs)`, `(coughs)`, `(sighs)`, `(gasps)`, `(clears throat)`, `(singing)`, `(sings)`, `(mumbles)`, `(beep)`, `(groans)`, `(sniffs)`, `(claps)`, `(screams)`, `(inhales)`, `(exhales)`, `(applause)`, `(burps)`, `(humming)`, `(sneezes)`, `(chuckle)`, `(whistles)` (full list, README §Features) | **No special handling at all.** These are ordinary text substrings that pass through `_encode_text` unchanged — each character (including the parentheses) becomes its own byte token. There is no dedicated non-verbal vocabulary, tag-parsing step, or auxiliary signal. See `architecture.md` §3 for the implication. |

`[S1]`/`[S2]` are the **only** two multi-character strings given
special-cased byte substitution. Everything else — punctuation, digits,
whitespace, non-verbal parentheticals, any language's script — is raw UTF-8
bytes with zero preprocessing.

## Pronunciation representation — investigated explicitly

Per the research brief, this is the most important question for Swara.
Checked every source file for any of: phoneme tables, G2P (grapheme-to-phoneme)
calls, IPA symbols, a pronunciation dictionary/lexicon file, stress markers,
or syllable boundaries.

**Finding: none exist.** Dia is **pure grapheme-level (byte-level) input with
implicit, learned pronunciation** — the model must infer English
pronunciation entirely from the statistical co-occurrence of byte sequences
and DAC audio-token sequences during training. There is:
- no `g2p` or `phonemizer` dependency in `pyproject.toml`
- no phoneme vocabulary in `config.py` (`vocab_size=256` is exactly and only
  the byte range)
- no dictionary/lexicon file anywhere in the repo tree
- no reference to IPA, ARPAbet, or any phoneme symbol set in any source file

**INFERENCE**: This architecture choice has direct, significant consequences
for Swara's stated goal of "strong Indian English pronunciation, Indian
names and place-name pronunciation":
1. Byte-level input means the model has *no explicit lever* to force a
   specific pronunciation for an ambiguous spelling (e.g. Indian proper
   nouns with non-obvious English pronunciation, like "Bengaluru",
   "Thiruvananthapuram", or common code-switched words). Any such control
   would have to come from either (a) training-data coverage of similar
   names, or (b) a Swara-side G2P/phoneme front-end that Dia's architecture
   simply does not have room for without modification.
2. It also means there is no tokenizer-training step to reproduce or
   depend on — byte-level vocab is trivially portable to any language or
   script without retraining an embedding table's *size* (though the actual
   embedding *weights* for untrained byte ranges would still need training
   exposure). This is a genuine advantage for eventual multilingual/code-
   switching support, since there's no fixed subword vocabulary to extend.
3. The `[S1]`/`[S2]` single-byte-substitution pattern is a reusable trick:
   any small set of structural control tokens can be injected into a
   byte-level vocabulary via reserved low byte values (`\x01`-`\x08` etc.,
   avoiding `\x00` which is the pad sentinel) without needing a "real"
   subword tokenizer.

See `swara-lessons.md` for how this maps to KEEP/RECONSIDER/AVOID categories.
