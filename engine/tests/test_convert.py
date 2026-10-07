from vaani_engine.convert_whisper import dims_from_config, mlx_name


def test_hf_whisper_names_map_to_mlx_layout():
    assert mlx_name("model.encoder.layers.3.self_attn.k_proj.weight") == "encoder.blocks.3.attn.key.weight"
    assert mlx_name("model.decoder.layers.0.encoder_attn.out_proj.bias") == "decoder.blocks.0.cross_attn.out.bias"
    assert mlx_name("model.decoder.layers.1.encoder_attn_layer_norm.weight") == "decoder.blocks.1.cross_attn_ln.weight"
    assert mlx_name("model.encoder.layers.2.fc2.weight") == "encoder.blocks.2.mlp2.weight"
    assert mlx_name("model.encoder.layers.2.final_layer_norm.bias") == "encoder.blocks.2.mlp_ln.bias"
    assert mlx_name("model.encoder.layer_norm.weight") == "encoder.ln_post.weight"
    assert mlx_name("model.decoder.layer_norm.bias") == "decoder.ln.bias"
    assert mlx_name("model.decoder.embed_positions.weight") == "decoder.positional_embedding"
    assert mlx_name("model.encoder.conv1.weight") == "encoder.conv1.weight"
    assert mlx_name("proj_out.weight") is None
    assert mlx_name("model.encoder.embed_positions.weight") is None


def test_dims_from_config():
    cfg = {"num_mel_bins": 80, "max_source_positions": 1500, "d_model": 1280, "encoder_attention_heads": 20,
           "encoder_layers": 32, "vocab_size": 51865, "max_target_positions": 448, "decoder_attention_heads": 20,
           "decoder_layers": 32}
    dims = dims_from_config(cfg)
    assert dims["n_text_layer"] == 32 and dims["n_mels"] == 80 and dims["model_type"] == "whisper"
