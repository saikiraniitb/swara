"""Bounded M2B architecture-validation run; no speech corpus or checkpoints."""

from __future__ import annotations

import argparse
import math

import torch

from swara.contracts import AudioTokenSpec, GenerationOptions, PronunciationInput, PronunciationOverride, build_plain_text_request
from swara.frontend import compile_request
from swara.models.generator import GeneratorConfig, LearnedSpeakerConditioner, SwaraSpeechGenerator
from swara.models.linguistic import LinguisticVocabulary
from swara.models.training import compute_token_losses


def _sequence(text: str, override: PronunciationOverride | None = None):
    request = build_plain_text_request(text, speaker_id="synthetic")
    if override is not None:
        request = request.__class__(request.content, request.speaker, PronunciationInput(overrides=(override,)))
    return compile_request(request)


def _synthetic_targets(example_index: int, frames: int = 4) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple((example_index * 191 + frame * 31 + codebook * 7 + 1) % 2048 for codebook in range(16)) for frame in range(frames))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-path", required=True, help="existing local Qwen tokenizer asset path")
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 500:
        raise ValueError("steps must be within 1..500")

    torch.manual_seed(20260822)
    torch.set_num_threads(1)
    spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
    sequences = (
        _sequence("Saikiran travelled.", PronunciationOverride(0, 8, "swara-phones-v0", ("S", "AI", "K", "I", "R", "A", "N"), "en-IN")),
        _sequence("Hyderabad is warm."),
        _sequence("Hello world!"),
    )
    vocabulary = LinguisticVocabulary.build(sequences)
    conditioner = LearnedSpeakerConditioner(("synthetic",))
    config = GeneratorConfig(vocabulary.size, 1, spec, model_dim=256, layers=4, heads=4, ffn_dim=512, max_text_tokens=32, max_audio_frames=4)
    model = SwaraSpeechGenerator(config, vocabulary, conditioner).train()
    encoded_rows = [vocabulary.encode(sequence).ids for sequence in sequences]
    # Pad the small synthetic batch to a regular tensor without introducing a second text tokenizer.
    max_text = max(len(row) for row in encoded_rows)
    padded_text = torch.zeros((len(sequences), max_text), dtype=torch.long)
    for index, row in enumerate(encoded_rows):
        padded_text[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    targets = torch.tensor([_synthetic_targets(index) for index in range(len(sequences))], dtype=torch.long)
    speakers = torch.zeros(len(sequences), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    initial_loss = None
    for step in range(1, args.steps + 1):
        inputs = model.teacher_forcing_inputs(targets)
        primary, residual, _ = model.forward(padded_text, speakers, inputs, primary_tokens_for_residual=targets[:, :, 0])
        losses = compute_token_losses(primary, residual, targets)
        if initial_loss is None:
            initial_loss = float(losses.total.detach())
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        inputs = model.teacher_forcing_inputs(targets)
        primary, residual, _ = model.forward(padded_text, speakers, inputs, primary_tokens_for_residual=targets[:, :, 0])
        losses = compute_token_losses(primary, residual, targets)
        predictions = torch.cat((primary.argmax(dim=-1).unsqueeze(-1), residual.argmax(dim=-1)), dim=-1)
        accuracy = float((predictions == targets).float().mean())
    if not (float(losses.total) < initial_loss * 0.2 and accuracy > 0.90):
        raise RuntimeError(f"synthetic overfit criterion failed: initial={initial_loss:.4f} final={float(losses.total):.4f} accuracy={accuracy:.4f}")
    print(f"OVERFIT steps={args.steps} initial_loss={initial_loss:.6f} final_loss={float(losses.total):.6f} token_accuracy={accuracy:.6f} parameters={model.parameter_count}")

    # Structural compatibility: actual codec encode confirms target geometry;
    # generated, learned tokens—not random IDs—are then sent through decode.
    from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
    from swara.contracts import AudioWaveform

    codec = Qwen12HzCodecAdapter.from_local_path(args.codec_path)
    if codec.spec != spec:
        raise RuntimeError("M2A codec AudioTokenSpec differs from the M2B generator spec")
    source = AudioWaveform(tuple(0.1 * math.sin(2 * math.pi * 220 * index / 24000) for index in range(6000)), 24000)
    real_tokens = codec.encode(source)
    real_tokens.validate_against(spec)
    generated = model.generate(sequences[0], conditioner.resolve("synthetic"), generation=GenerationOptions(deterministic=True, max_duration_ms=320))
    generated.validate_against(spec)
    decoded = codec.decode(generated)
    if not decoded.samples or not all(math.isfinite(sample) for sample in decoded.samples):
        raise RuntimeError("codec produced an invalid waveform from generated tokens")
    print(f"STRUCTURAL generated_shape=({len(generated.frames)},{len(generated.frames[0])}) real_encoded_shape=({len(real_tokens.frames)},{len(real_tokens.frames[0])}) waveform_samples={len(decoded.samples)} sample_rate={decoded.sample_rate_hz}")


if __name__ == "__main__":
    main()
