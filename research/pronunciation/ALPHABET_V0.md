# Pronunciation Alphabet v0

**Pronunciation Alphabet v0 is an architecture/testing alphabet only.** It is not a final Indian phoneme inventory, IPA substitute, pronunciation guide, or claim of correct pronunciation for any fixture.

## Contract

- Identifier: `swara-phones-v0`
- Container: an ordered tuple of uppercase human-readable symbols.
- Validation: every symbol must be in the finite set below.
- Scope: explicit override spans only. Untouched text remains grapheme tokens.
- Extension: a future alphabet receives a new identifier; it must not silently change this set.

## Allowed symbols

```text
A AA E EE I II O OO U UU AI AU
K G T D N P B M Y R L V S H SH CH J NG
```

The set is deliberately small: vowel-like units and common consonant-like units are enough to prove typed override spans, validation, language metadata, and future embedding boundaries. It omits stress, tone, aspiration detail, schwa policy, Indian-language phoneme coverage, and many English distinctions.

## Example only

`Saikiran` may be represented in a test fixture as `(S, AI, K, I, R, A, N)`. This verifies an explicit override pipeline; it does not assert a normative pronunciation.

