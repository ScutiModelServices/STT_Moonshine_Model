# Purpose

**Kind:** audio → text

**Group:** Speech-to-Text

**Tags:** `cpu`, `streaming`, `variants`

| | |
|---|---|
| Input | Raw 16 kHz mono signed 16-bit PCM (no other formats; caller decodes). Options: beam size, stream. |
| Output | JSON: text, segments, duration. English only, no word timestamps. |
| Runs on | CPU (Moonshine ONNX, `base` default). The Model variant card lists size x number format with an estimate of the memory each takes and whether this machine can run it. |

**Use for:** English transcription, alternative to Whisper on the same API. Model half of `ToolServices/Scribe`.
**Not for:** other languages, word timestamps, audio decoding.
