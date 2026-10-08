# Third-party software, models and services

Vaani Studio's own code is MIT-licensed (see [LICENSE](LICENSE)). It builds on the work below.
The app download contains only Vaani's code and the `uv` installer. Everything else is downloaded
onto your Mac by the app: Python packages during the one-time setup, and AI models the first time
each feature is used. Each item stays under its own license.

## Included in the app download

| Component | License |
|---|---|
| [uv](https://github.com/astral-sh/uv) (Python installer) | MIT or Apache-2.0 |

## Python packages installed during setup (main ones)

| Package | License |
|---|---|
| [PyTorch](https://pytorch.org) | BSD-3-Clause |
| [MLX](https://github.com/ml-explore/mlx), mlx-whisper, mlx-lm | MIT |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) | MIT |
| [Hugging Face Transformers](https://github.com/huggingface/transformers), huggingface_hub | Apache-2.0 |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT / Apache-2.0 |
| [Demucs](https://github.com/facebookresearch/demucs) | MIT |
| [Pedalboard](https://github.com/spotify/pedalboard) (Spotify) | **GPL-3.0** |
| [PyAV](https://github.com/PyAV-Org/PyAV) (bundles FFmpeg libraries) | BSD-3-Clause (FFmpeg: LGPL/GPL) |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT |
| [Chatterbox](https://github.com/resemble-ai/chatterbox), resemble-perth (Resemble AI) | MIT |
| [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) | MIT |
| [FastAPI](https://fastapi.tiangolo.com), Uvicorn, Pydantic, httpx | MIT / BSD |
| noisereduce, pyloudnorm, soundfile, SciPy, NumPy | MIT / BSD |
| anthropic (Python SDK) | MIT |
| openai (Python SDK) | Apache-2.0 |
| webrtcvad-wheels | MIT |

## AI models downloaded on first use

| Model | Used for | License |
|---|---|---|
| Whisper large-v3 turbo (`mlx-community/whisper-large-v3-turbo`) | Transcription | MIT |
| Whisper large-v2 Marathi (`DrishtiSharma/whisper-large-v2-marathi`), converted to MLX on your Mac | Marathi transcription | Apache-2.0 |
| MossFormer2 SE 48K (`starkdmi/MossFormer2-SE`) | Noise removal | Apache-2.0 |
| DeepFilterNet 3, htdemucs | Noise removal, voice separation | MIT |
| Gemma 3 12B (`mlx-community/gemma-3-12b-it-4bit`) | On-device translation | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| Qwen3 4B Instruct (`mlx-community/Qwen3-4B-Instruct-2507-4bit`) | Hinglish, lightweight translation | Apache-2.0 |
| Chatterbox multilingual (`mlx-community/chatterbox-multilingual-v3`, `ResembleAI/chatterbox`) | Voice cloning | MIT |
| MMS-TTS Marathi (`facebook/mms-tts-mar`) | Marathi speech | **CC-BY-NC-4.0 (non-commercial)** |
| IndicConformer, IndicTrans2 (AI4Bharat). Optional, gated: you accept their terms on Hugging Face | Marathi/Hindi transcription, translation | MIT |

## Cloud services (optional, with your own API keys)

[Sarvam AI](https://www.sarvam.ai), [Anthropic](https://www.anthropic.com) and [OpenAI](https://openai.com).
Their own terms of service and pricing apply. Audio or text is sent to a service only when a job
uses it, and the app shows exactly which ones before you start.

## Voice cloning

Only clone voices of people who have agreed to it. Chatterbox output carries Resemble AI's
inaudible Perth watermark that marks it as AI-generated.
