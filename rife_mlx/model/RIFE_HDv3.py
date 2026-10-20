"""Model wrapper — MLX port of train_log/RIFE_HDv3.py.

Wraps IFNet and exposes inference(img0, img1, timestep, scale) -> middle frame,
matching upstream: scale_list = [16,8,4,2,1] / scale; H,W padded to a multiple of
pad_to (64; the 5-block downsample factor) and cropped back after.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .IFNet_HDv3 import IFNet


@dataclass(frozen=True)
class PreparedFrame:
    """A padded input and its reusable image features."""

    image: mx.array
    features: mx.array
    height: int
    width: int
    scale: float


class Model:
    def __init__(self, scale_list=(16, 8, 4, 2, 1), pad_to: int = 64) -> None:
        self.flownet = IFNet()
        self.scale_list = tuple(scale_list)
        self.pad_to = pad_to

    def prepare(self, image: mx.array, scale: float = 1.0) -> PreparedFrame:
        """Pad and encode one input frame, materialising reusable state."""
        _batch, height, width, _channels = image.shape
        pad_to = max(self.pad_to, int(round(self.pad_to / scale)))
        padded_height = ((height - 1) // pad_to + 1) * pad_to
        padded_width = ((width - 1) // pad_to + 1) * pad_to
        padding = [
            (0, 0),
            (0, padded_height - height),
            (0, padded_width - width),
            (0, 0),
        ]
        padded = mx.pad(image, padding)
        features = self.flownet.encode(padded[..., :3])
        mx.eval(padded, features)
        return PreparedFrame(padded, features, height, width, scale)

    def inference_prepared(self, frame0: PreparedFrame, frame1: PreparedFrame,
                           timestep: float = 0.5) -> mx.array:
        """Interpolate frames prepared with the same dimensions and scale."""
        if (frame0.height, frame0.width, frame0.scale) != (
                frame1.height, frame1.width, frame1.scale):
            raise ValueError("prepared frames must have the same size and scale")
        if frame0.image.shape != frame1.image.shape:
            raise ValueError("prepared frames must have matching padded shapes")

        scale_list = tuple(s / frame0.scale for s in self.scale_list)
        inputs = mx.concatenate([frame0.image, frame1.image], axis=-1)
        _flow, _mask, merged = self.flownet(
            inputs, timestep, scale_list,
            f0=frame0.features, f1=frame1.features)
        return merged[:, :frame0.height, :frame0.width, :]

    def inference(self, img0: mx.array, img1: mx.array,
                  timestep: float = 0.5, scale: float = 1.0) -> mx.array:
        """img0,img1: [N,H,W,3] in [0,1]. Returns the frame at `timestep`."""
        N, H, W, _ = img0.shape
        # pad must scale with 1/scale so the coarsest-scale interpolation
        # round-trips exactly (else the recovered flow size != padded size).
        pad = max(self.pad_to, int(round(self.pad_to / scale)))
        ph = ((H - 1) // pad + 1) * pad
        pw = ((W - 1) // pad + 1) * pad
        pad = [(0, 0), (0, ph - H), (0, pw - W), (0, 0)]
        i0 = mx.pad(img0, pad)
        i1 = mx.pad(img1, pad)

        scale_list = tuple(s / scale for s in self.scale_list)
        x = mx.concatenate([i0, i1], axis=-1)
        _flow, _mask, merged = self.flownet(x, timestep, scale_list)
        return merged[:, :H, :W, :]
