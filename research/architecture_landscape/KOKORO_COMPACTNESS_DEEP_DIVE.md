# Kokoro compactness and architecture

Source: `hexgrad/kokoro`, commit `dfb907a02bba8152ca444717ca5d78747ccb4bec`; files `kokoro/model.py`, `modules.py`, `istftnet.py`, `pipeline.py`, `README.md`.

## Model path

Kokoro-82M is an 82M-parameter StyleTTS2-derived acoustic model. `KPipeline` uses Misaki G2P (with eSpeak fallback) and language codes; `KModel.forward_with_tokens` embeds phoneme IDs with a frozen/loaded ALBERT (`CustomAlbert`), projects to `hidden_dim`, then uses `ProsodyPredictor`. The predictor has a style-conditioned duration path, bidirectional LSTM, F0/noise prediction, and explicit monotonic alignment built from rounded per-token durations. `TextEncoder` is a convolution stack plus bidirectional LSTM. The decoder is an iSTFTNet neural vocoder with AdaIN style conditioning, F0 harmonic/noise paths and learned STFT inversion; output is 24 kHz.

Tensor outline: phoneme IDs `(B,N)` → ALBERT `(B,N,H_bert)` → `bert_encoder` `(B,N,H)`; duration predictor → integer `pred_dur[N]`; alignment matrix `(B,N,T)`; text encoder and prosody expand to frame states `(B,H,T)`; iSTFTNet produces waveform. `style_dim` is 128 in `forward_with_tokens` (`ref_s[:,128:]` is prosody style and `ref_s[:,:128]` vocoder style in the voice pack). Exact config dimensions are checkpoint-config dependent and not embedded in this source-only clone.

## Frontend and controls

Misaki supports English, Spanish, French, Hindi, Italian, Japanese, Mandarin and Portuguese routes. The English path includes phoneme substitutions, punctuation-preserving tokenization, chunking and optional eSpeak fallback. Voice packs are learned style tensors; they are cacheable but are not an arbitrary reference-audio encoder in this API. Speed modifies predicted durations. There are no structured emotion/energy controls in the public inference API; style is primarily voice-pack conditioning.

## Why no residual codec

Kokoro predicts duration/F0/noise and mel/STFT-like acoustic features directly, then synthesizes waveform with iSTFTNet. There is no discrete multi-codebook codec or residual token chain. The alignment matrix is the explicit mechanism that prevents text/audio schedule drift. This is a useful contrast to Swara's failed frame schedule, but the StyleTTS2 training stack is not a direct token-generator replacement.

Evidence: 82M and Apache weights are README claims; module graph and duration/alignment operations are CONFIRMED FROM SOURCE. Dataset, adversarial/discriminator losses and exact parameter allocation are not present in this inference repository.
