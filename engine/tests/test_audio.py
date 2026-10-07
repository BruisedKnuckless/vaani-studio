import numpy as np

from vaani_engine import audio_io
from vaani_engine.pipelines import dsp

SR = 48000


def tone(seconds=3.0, freq=220.0, amp=0.3, channels=2):
    t = np.arange(int(seconds * SR)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x] * channels)


def test_wav_round_trip_keeps_shape_and_rate(tmp_path):
    x = tone()
    path = audio_io.save(tmp_path / "t.wav", x, SR)
    y = audio_io.load(path, sr=SR)
    assert y.shape == x.shape
    assert np.allclose(x, y, atol=1e-4)
    info = audio_io.probe(path)
    assert info.sample_rate == SR and info.channels == 2


def test_compressed_formats_encode_and_decode(tmp_path):
    x = tone(channels=1)
    for fmt in ("m4a", "mp3", "flac"):
        path = audio_io.save(tmp_path / "t", x, SR, fmt)
        assert path.suffix == f".{fmt}"
        y = audio_io.load(path, sr=16000, channels=1)
        assert abs(y.shape[-1] / 16000 - 3.0) < 0.1


def test_resample_length():
    y = audio_io.resample(tone(channels=1), SR, 16000)
    assert y.shape == (1, 16000 * 3)


def test_normalize_hits_target_and_respects_peak_ceiling():
    x = tone(amp=0.05) + np.random.default_rng(0).normal(0, 0.01, (2, 3 * SR)).astype(np.float32)
    x[:, SR:SR + 100] = 0.9  # a spike the limiter has to catch
    y = dsp.normalize(x, SR, -16.0)
    assert abs(dsp.loudness(y, SR) - (-16.0)) < 0.5
    assert dsp.peak_db(y) <= dsp.TRUE_PEAK_DB - 0.49


def test_peak_limit_leaves_quiet_audio_untouched():
    x = tone(amp=0.1)
    assert np.array_equal(dsp.peak_limit(x, SR), x)


def test_de_esser_is_transparent_below_threshold():
    x = tone(amp=0.01, freq=7000.0)
    assert np.allclose(dsp.de_ess(x, SR), x, atol=1e-5)


def test_peaks_bucket_count():
    p = audio_io.peaks(tone(), 100)
    assert len(p) == 100 and max(p) <= 0.31
