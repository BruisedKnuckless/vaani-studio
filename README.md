# Vaani Studio

A Mac app for speech in **Marathi, Hindi, English and Hinglish**:

- **Enhance**: remove background noise, isolate the voice from music, even out the voice and set the loudness.
- **Transcribe & Translate**: get editable transcripts and subtitles (SRT, VTT, TXT, JSON), then translate them.
- **Dub**: Marathi ↔ English and Hindi ↔ English, spoken in the original speaker's voice.

Every step can run **on this Mac** or through **cloud APIs** (Sarvam AI, Anthropic Claude, OpenAI). You choose each time you add a file: **On this Mac**, **Cloud API** or **Auto**. Before you start, the app shows exactly which service will handle each step.

## Setup

Requirements: an Apple Silicon Mac, Homebrew and Xcode.

```bash
scripts/bootstrap.sh     # tools, Python 3.11 engine, voice runtimes (~4 GB of packages)
scripts/build_app.sh     # builds build/Vaani.app
open build/Vaani.app
```

Models download on first use, or ahead of time from the **Models** screen. They are stored in `~/Library/Application Support/VaaniStudio/models`. Results are saved to `~/Music/VaaniStudio/<job>/`. You can change that folder in Settings.

### API keys (optional)

API keys go in **Settings**, which saves them in the macOS Keychain.

| Key | Used for |
|---|---|
| Sarvam AI | Best Marathi/Hindi/Hinglish transcription, Sarvam-105B translation (natural, meaning-based), voice-cloned speech and studio voices in English, Hindi and Marathi |
| Anthropic | Claude translation: context-aware, Hinglish, shortened lines for dubbing |
| OpenAI | gpt-4o-transcribe and GPT translation |
| Hugging Face (free) | Unlocks AI4Bharat's gated models: IndicConformer (local Marathi transcription) and IndicTrans2 (local translation). Accept each model's terms on huggingface.co first. |

With no keys, everything still runs locally.

## What runs where

| Step | On this Mac | Cloud |
|---|---|---|
| Voice/background separation | Demucs htdemucs | (none) |
| Noise removal | DeepFilterNet 3 (fast), MossFormer2 48K (high quality, MLX), spectral gate | (none) |
| Polish | EQ, de-esser, compressor, loudness normalisation with a look-ahead peak limiter | (none) |
| Transcription | Whisper large-v3 turbo (MLX); IndicConformer 600M for Marathi (needs HF token) | Sarvam Saaras v4 (`translit` mode for Hinglish), OpenAI gpt-4o-transcribe |
| Translation | Gemma 3 12B (MLX), IndicTrans2 (needs HF token), Qwen3 4B for Hinglish | Claude, OpenAI, Sarvam-105B, Sarvam Mayura (literal) |
| Voice-cloned speech | Chatterbox multilingual (MLX) for English and Hindi; MMS-TTS + Chatterbox voice conversion for Marathi | Sarvam voice cloning (Bulbul), or Sarvam studio voices |

**Auto** mode prefers:
- Sarvam for Marathi and Hinglish transcription
- an LLM for translation
- local Chatterbox for English and Hindi voices
- Sarvam for Marathi voices, when a key is set

If a step fails in Auto, it falls back to the other side. In On this Mac or Cloud API mode, a failure is reported instead, with a **Run this step locally** button.

## Command line

```bash
cd engine
.venv/bin/vaani enhance "talk.wav" --quality high --isolate
.venv/bin/vaani transcribe "talk.wav" --lang mr --to en --mode auto
.venv/bin/vaani dub "talk.wav" --from mr --to en --mode local
.venv/bin/vaani plan dub --from en --to mr --mode cloud   # show which service would run each step
.venv/bin/vaani models list
.venv/bin/python -m pytest
```

## Layout

```
app/                     SwiftUI app (XcodeGen project.yml)
engine/vaani_engine/     Python engine: FastAPI server, job queue, pipelines, providers
engine/runtimes/         Isolated environments for model families with conflicting dependencies
  mlxvoice/              Chatterbox on MLX (voice cloning)
  voice/                 Chatterbox VC + MMS-TTS (PyTorch)
scripts/                 bootstrap.sh, build_app.sh
```

The app starts the engine on `127.0.0.1` at a random port, with a per-launch token. API keys are passed in memory and never written to disk.

## Notes

- **Licences.** Chatterbox, Whisper, DeepFilterNet, Demucs, MossFormer2, IndicConformer and IndicTrans2 are MIT or Apache-2.0. MMS-TTS is CC-BY-NC-4.0, which allows non-commercial use only.
- **Watermark.** Chatterbox output carries Resemble AI's inaudible Perth watermark that marks it as AI-generated.
- **Consent.** Only clone the voices of people who have agreed to it.
