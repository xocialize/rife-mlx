"""P5 pipeline gates — needs dist/ weights. Skips if absent.

  - interpolate_sequence frame-count math: (N-1)*multi + 1
  - t-sweep: t=0.25/0.5/0.75 produce distinct frames, motion monotone in t
    (a vertical bar shifted between img0 and img1 moves with t).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from rife_mlx import pipeline_mlx
from rife_mlx.pipeline_mlx import interpolate_pair, interpolate_sequence, interpolate_stream
from rife_mlx.utils.weights import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST = os.path.join(ROOT, "dist", "RIFE-4.25")


@pytest.fixture(scope="module")
def model():
    if not os.path.isdir(DIST):
        pytest.skip("missing dist/RIFE-4.25")
    return build_model("4.25", weights_dir=DIST)


def _bar_frame(x):
    img = np.zeros((64, 96, 3), np.uint8)
    img[:, max(0, x - 4):x + 4, :] = 255
    return img


def test_sequence_frame_count(model):
    frames = [_bar_frame(8), _bar_frame(40), _bar_frame(72)]  # 3 frames
    for multi in (2, 3, 4):
        out = interpolate_sequence(model, frames, multi=multi)
        assert len(out) == (len(frames) - 1) * multi + 1


def test_interpolate_stream_is_lazy_and_preserves_order(monkeypatch):
    consumed = []

    def source():
        for value in (0, 10, 20):
            consumed.append(value)
            yield np.full((2, 3, 3), value, dtype=np.uint8)

    class FakeModel:
        def prepare(self, frame, scale):
            assert scale == 1.0
            return frame

        def inference_prepared(self, a, b, timestep):
            return a * (1 - timestep) + b * timestep

    output = interpolate_stream(FakeModel(), source(), multi=2)

    first = next(output)
    assert int(first[0, 0, 0]) == 0
    assert consumed == [0, 10]
    values = [int(frame[0, 0, 0]) for frame in output]
    assert values == [5, 10, 15, 20]
    assert consumed == [0, 10, 20]


def test_interpolate_stream_handles_empty_and_single_frame():
    assert list(interpolate_stream(None, iter(()))) == []
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    assert list(interpolate_stream(None, iter((frame,)))) == [frame]
    with pytest.raises(ValueError, match="multi must be >= 1"):
        next(interpolate_stream(None, (), multi=0))


def test_interpolate_stream_caches_each_input_once(monkeypatch):
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (0, 20, 40)]

    class FakeModel:
        def __init__(self):
            self.prepared = []

        def prepare(self, frame, scale):
            self.prepared.append((frame, scale))
            return frame

        def inference_prepared(self, frame0, frame1, timestep):
            return frame0 * (1 - timestep) + frame1 * timestep

    monkeypatch.setattr(pipeline_mlx, "_to_nhwc", lambda frame: frame.astype(float))
    monkeypatch.setattr(pipeline_mlx, "_to_u8", lambda frame: frame.astype(np.uint8))
    monkeypatch.setattr(pipeline_mlx.mx, "eval", lambda *args: None)
    model = FakeModel()

    output = list(interpolate_stream(model, iter(frames), multi=4, scale=0.5))
    assert [int(frame[0, 0, 0]) for frame in output] == [0, 5, 10, 15, 20, 25, 30, 35, 40]
    assert len(model.prepared) == len(frames)
    assert all(scale == 0.5 for _frame, scale in model.prepared)


def test_interpolate_stream_multi_one_bypasses_model():
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (0, 20, 40)]
    output = list(interpolate_stream(None, iter(frames), multi=1))
    assert output == frames


@pytest.mark.parametrize("H,W,scale", [
    (1080, 1920, 0.5),   # regression: 1080 pads to 1088 at /64 but needs 1152 at scale 0.5
    (1080, 1920, 0.25),
    (721, 1281, 0.5),    # odd dims
    (240, 320, 1.0),
])
def test_scale_padding_roundtrip(model, H, W, scale):
    rng = np.random.default_rng(0)
    a = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    b = np.roll(a, 4, axis=1)
    out = interpolate_pair(model, a, b, 0.5, scale)
    assert out.shape == (H, W, 3)


def test_t_sweep_monotone_motion(model):
    img0, img1 = _bar_frame(16), _bar_frame(72)  # bar moves left->right
    def bar_x(frame):  # centroid column of the bright bar
        col = frame.mean(axis=(0, 2))
        return float((np.arange(len(col)) * col).sum() / (col.sum() + 1e-6))
    xs = [bar_x(interpolate_pair(model, img0, img1, t)) for t in (0.25, 0.5, 0.75)]
    print(f"\nbar centroid by t: {[round(x,1) for x in xs]}")
    assert xs[0] < xs[1] < xs[2], f"motion not monotone in t: {xs}"
