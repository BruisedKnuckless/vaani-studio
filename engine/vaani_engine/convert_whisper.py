"""Convert a Hugging Face (transformers) Whisper checkpoint to the MLX layout used by
mlx-whisper, so fine-tuned models run as fast as the stock ones on Apple Silicon."""

from __future__ import annotations

import json
from pathlib import Path

RENAMES = [
    (".self_attn_layer_norm.", ".attn_ln."),
    (".encoder_attn_layer_norm.", ".cross_attn_ln."),
    (".final_layer_norm.", ".mlp_ln."),
    (".self_attn.k_proj.", ".attn.key."),
    (".self_attn.q_proj.", ".attn.query."),
    (".self_attn.v_proj.", ".attn.value."),
    (".self_attn.out_proj.", ".attn.out."),
    (".encoder_attn.k_proj.", ".cross_attn.key."),
    (".encoder_attn.q_proj.", ".cross_attn.query."),
    (".encoder_attn.v_proj.", ".cross_attn.value."),
    (".encoder_attn.out_proj.", ".cross_attn.out."),
    (".fc1.", ".mlp1."),
    (".fc2.", ".mlp2."),
    (".layers.", ".blocks."),
]
EXACT = {
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
}
SKIP = {"proj_out.weight", "encoder.embed_positions.weight"}  # tied output layer; sinusoidal encoder positions


def mlx_name(hf_name: str) -> str | None:
    name = hf_name.removeprefix("model.")
    if name in SKIP:
        return None
    if name in EXACT:
        return EXACT[name]
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def dims_from_config(cfg: dict) -> dict:
    return {
        "n_mels": cfg["num_mel_bins"],
        "n_audio_ctx": cfg["max_source_positions"],
        "n_audio_state": cfg["d_model"],
        "n_audio_head": cfg["encoder_attention_heads"],
        "n_audio_layer": cfg["encoder_layers"],
        "n_vocab": cfg["vocab_size"],
        "n_text_ctx": cfg["max_target_positions"],
        "n_text_state": cfg["d_model"],
        "n_text_head": cfg["decoder_attention_heads"],
        "n_text_layer": cfg["decoder_layers"],
        "model_type": "whisper",
    }


def convert(hf_dir: Path, out_dir: Path | None = None, remove_source: bool = False) -> Path:
    import mlx.core as mx
    import numpy as np
    import torch

    hf_dir = Path(hf_dir)
    out_dir = Path(out_dir or hf_dir)
    cfg = json.loads((hf_dir / "config.json").read_text())
    safetensors = hf_dir / "model.safetensors"
    if safetensors.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors))
    else:
        state = torch.load(hf_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)

    weights = {}
    for hf_name, tensor in state.items():
        name = mlx_name(hf_name)
        if name is None:
            continue
        array = tensor.to(torch.float16).numpy()
        if name.endswith(("conv1.weight", "conv2.weight")):
            array = np.ascontiguousarray(array.transpose(0, 2, 1))  # PyTorch [out, in, k] → MLX [out, k, in]
        weights[name] = mx.array(array)
    del state

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "weights.safetensors"), weights)
    (out_dir / "config.json").write_text(json.dumps(dims_from_config(cfg), indent=2))
    if remove_source:  # the MLX weights live in weights.safetensors, so the originals can go
        for leftover in ("pytorch_model.bin", "model.safetensors"):
            (hf_dir / leftover).unlink(missing_ok=True)
    return out_dir
