# Swara

Swara currently provides framework-neutral contracts and an M1 deterministic linguistic frontend. It does not yet synthesize audio.

```python
from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.frontend import compile_request

text = "Saikiran travelled from Hyderabad."
request = SynthesisRequest(
    content=Content(text=text, default_language="en-IN"),
    speaker=SpeakerRef("default"),
    pronunciation=PronunciationInput(
        overrides=(
            PronunciationOverride(
                start=0,
                end=8,
                pronunciation_system="swara-phones-v0",
                tokens=("S", "AI", "K", "I", "R", "A", "N"),
                language="en-IN",
            ),
        )
    ),
)

sequence = compile_request(request)
```

All caller-supplied ranges use Python Unicode string/code-point offsets into the original source text. The frontend internally projects them through conservative normalization.

