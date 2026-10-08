# Vaani Studio

A free, open-source Mac app for speech in **Marathi, Hindi, English and Hinglish**:

- **Enhance**: remove background noise, separate a voice from music, and get clear, evenly levelled speech.
- **Transcribe & Translate**: get editable transcripts and subtitles (SRT, VTT, TXT, JSON) and translate them.
- **Dub**: Marathi ↔ English and Hindi ↔ English. Whole sentences are voiced at a natural pace, either in the
  **original speaker's own voice** (voice cloning) or in a **studio voice you choose and preview**.

Each step runs either **on your Mac** (private and free) or through **cloud APIs** (Sarvam AI, Anthropic Claude,
OpenAI) with your own keys. You choose for every file: *On this Mac*, *Cloud API* or *Auto*. Before you start,
the app shows exactly which service will handle each step and whether anything will be uploaded.

## Install

1. Download the latest `VaaniStudio-x.y.z.dmg` from [Releases](../../releases).
2. Open it and drag **Vaani** to **Applications**.
3. Open Vaani. It isn't notarised by Apple, so the first time macOS says it "could not verify" the app:
   1. Click **Done**.
   2. Open **System Settings → Privacy & Security**.
   3. Click **Open Anyway** next to the Vaani message, then confirm.

   You only do this once.
4. Click **Install the Engine**. This is a one-time download of about 2 GB (about 4 GB on disk). AI models download
   the first time you use each feature.

**Requirements:**
- an Apple Silicon Mac (M1 or later) with macOS 14 or later
- an internet connection for setup
- about 15 GB of free space for the engine and all the models

Intel Macs aren't supported.

### Optional API keys

API keys go in **Settings** and are stored in the macOS Keychain. With no keys, everything runs on your Mac.

| Key | What it adds |
|---|---|
| [Sarvam AI](https://dashboard.sarvam.ai) | The most accurate Marathi, Hindi and Hinglish transcription; Sarvam-105B translation; cloned and studio voices in English, Hindi and Marathi |
| [Anthropic](https://console.anthropic.com) | Claude translation, the most natural English |
| [OpenAI](https://platform.openai.com) | Transcription and translation |
| [Hugging Face](https://huggingface.co/settings/tokens) (free) | Optional on-device models from AI4Bharat: IndicConformer and IndicTrans2. Accept each model's terms on its page first. |

## What runs where

| Step | On your Mac | Cloud |
|---|---|---|
| Noise removal | DeepFilterNet 3 (fast), MossFormer2 48K (high quality) | |
| Voice/music separation | Demucs | |
| Transcription | Whisper large-v3 turbo; Whisper large-v2 fine-tuned for Marathi; IndicConformer (optional) | Sarvam Saaras, OpenAI |
| Translation | Gemma 3 12B; Qwen3 4B for Hinglish; IndicTrans2 (optional) | Claude, Sarvam-105B, OpenAI, Sarvam Mayura |
| Dubbing voice | Chatterbox (voice cloning, English and Hindi); MMS-TTS with voice conversion (Marathi) | Sarvam voice cloning; Sarvam studio voices |
| Finish | EQ, de-esser, compression, loudness levelling to −16 LUFS | |

All on-device AI runs on Apple Silicon. Most of it uses [MLX](https://github.com/ml-explore/mlx), Apple's machine-learning framework. Files are saved to
`~/Music/VaaniStudio`, which you can change in Settings.

## Privacy

- **On this Mac:** nothing leaves your computer.
- **Cloud steps:** audio or text goes only to the service shown in the plan.
- **API keys:** kept in your Keychain and never written to files.

The engine listens only on `127.0.0.1`, with a fresh access token each launch.

## Build from source

You need an Apple Silicon Mac with Xcode and [Homebrew](https://brew.sh).

```bash
git clone https://github.com/BruisedKnuckless/vaani-studio.git && cd vaani-studio
scripts/bootstrap.sh            # xcodegen + uv, Python 3.11 engine and voice runtimes, then runs the tests
scripts/build_app.sh Debug      # dev build that runs the engine straight from engine/
open build/Vaani.app
scripts/make_dmg.sh             # shareable dist/VaaniStudio-<version>.dmg
```

The engine also has a command line:

```bash
cd engine
.venv/bin/vaani enhance talk.wav --quality high
.venv/bin/vaani transcribe talk.wav --lang mr --to en --mode auto
.venv/bin/vaani dub talk.wav --from mr --to en --voice sarvam_studio --voice-name ishita
.venv/bin/vaani plan dub --from en --to mr --mode cloud   # which service would run each step
.venv/bin/python -m pytest
```

### Layout

```
app/                     SwiftUI app (XcodeGen: app/project.yml)
engine/vaani_engine/     Python engine: FastAPI server, job queue, pipelines, providers
engine/runtimes/         Isolated environments for model families with conflicting dependencies
  mlxvoice/              Chatterbox on MLX (voice cloning)
  voice/                 Chatterbox voice conversion + MMS-TTS (PyTorch)
scripts/                 bootstrap.sh, build_app.sh, make_dmg.sh
```

The app runs the engine as a local process. Shared builds carry the engine's source, its lockfiles and `uv`
inside the app bundle. On first launch they install Python 3.11 and the engine into
`~/Library/Application Support/VaaniStudio`, so neither Homebrew nor Xcode is needed.

## Responsible use

Only clone the voices of people who have agreed to it. Voices cloned on your Mac carry Resemble AI's inaudible
watermark that marks them as AI-generated. Some models have their own licenses; MMS-TTS Marathi, for example, is
non-commercial. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

[MIT](LICENSE) © 2026 Viraj Wanjari. Third-party components keep their own licenses.
