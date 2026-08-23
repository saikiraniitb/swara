# M4A pretrained foundation baseline

M4A establishes Qwen3-TTS 0.6B Base as Swara's inference bootstrap. The
randomly initialized Swara v0/v1/v2 experiments were intentionally closed:
small-utterance teacher-forcing memorization did not produce reliable
free-running text control. A proven pretrained foundation is therefore the
correct next engineering boundary.

The Qwen model remains an external Apache-2.0 pretrained asset and is not the
final Swara engine. The adapter is original Swara code and loads Qwen lazily.
Swara retains ownership of the pronunciation frontend, structured controls,
speaker contract, long-form orchestration, and future reengineering,
adaptation, distillation, and efficiency work.
