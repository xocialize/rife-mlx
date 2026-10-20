"""High-level interpolation + CLI (image-pair + full video). P5.

    rife-mlx -i in.mp4 -o out.mp4 --multi 2 --scale 1.0
    rife-mlx --img0 a.png --img1 b.png -t 0.5 -o mid.png

--multi N inserts N-1 frames between each pair (at t = i/N); output fps = N*input.
--scale {0.25,0.5,1,2,4} sets the coarse-to-fine pyramid scale (0.5 for 4K).
"""

from __future__ import annotations

import argparse
import os

import mlx.core as mx
import numpy as np

from .config import DEFAULT_VERSION
from .utils.weights import build_model


def _to_nhwc(img_u8: np.ndarray) -> mx.array:
    return mx.array(img_u8.astype(np.float32)[None] / 255.0)


def _to_u8(arr: mx.array) -> np.ndarray:
    a = np.clip(np.array(arr)[0], 0.0, 1.0)
    return (a * 255.0).round().astype(np.uint8)


def interpolate_pair(model, img0_u8, img1_u8, timestep=0.5, scale=1.0) -> np.ndarray:
    out = model.inference(_to_nhwc(img0_u8), _to_nhwc(img1_u8), timestep, scale)
    mx.eval(out)
    return _to_u8(out)


def interpolate_sequence(model, frames, multi=2, scale=1.0):
    """Collect the cached interpolation stream into a list."""
    return list(interpolate_stream(model, frames, multi, scale))


def interpolate_stream(model, frames, multi=2, scale=1.0):
    """Yield RGB frames, preparing each input once and retaining the current pair."""
    if multi < 1:
        raise ValueError("multi must be >= 1")

    iterator = iter(frames)
    try:
        previous = next(iterator)
    except StopIteration:
        return

    if multi == 1:
        yield previous
        yield from iterator
        return

    previous_prepared = None
    for current in iterator:
        if previous_prepared is None:
            previous_prepared = model.prepare(_to_nhwc(previous), scale)
        current_prepared = model.prepare(_to_nhwc(current), scale)
        yield previous
        for j in range(1, multi):
            out = model.inference_prepared(previous_prepared, current_prepared, j / multi)
            mx.eval(out)
            yield _to_u8(out)
        previous = current
        previous_prepared = current_prepared

    yield previous


def cli_main() -> None:
    p = argparse.ArgumentParser(description="Practical-RIFE 4.25 MLX frame interpolation")
    p.add_argument("-i", "--input", help="input video")
    p.add_argument("--img0"); p.add_argument("--img1")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-t", "--timestep", type=float, default=0.5, help="image-pair mode")
    p.add_argument("-m", "--multi", type=int, default=2, help="video fps multiplier")
    p.add_argument("-s", "--scale", type=float, default=1.0,
                   choices=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("-n", "--version", default=DEFAULT_VERSION)
    p.add_argument("--weights_dir", default=None)
    args = p.parse_args()

    if bool(args.img0) != bool(args.img1):
        p.error("image mode requires both --img0 and --img1")
    if not ((args.img0 and args.img1) or args.input):
        p.error("give -i/--input, or both --img0 and --img1")
    if args.multi < 1:
        p.error("--multi must be >= 1")
    for source in (args.input, args.img0, args.img1):
        if source and os.path.exists(args.output) and os.path.samefile(source, args.output):
            p.error(f"output {args.output} is the same file as input {source}")

    model = build_model(args.version, weights_dir=args.weights_dir)

    if args.img0 and args.img1:
        from PIL import Image
        a = np.asarray(Image.open(args.img0).convert("RGB"))
        b = np.asarray(Image.open(args.img1).convert("RGB"))
        mid = interpolate_pair(model, a, b, args.timestep, args.scale)
        Image.fromarray(mid).save(args.output)
        print(f"{args.img0} + {args.img1} @t={args.timestep} -> {args.output} {mid.shape}")
        return

    from .video import VideoWriter, iter_frames, probe_fps
    fps = probe_fps(args.input)
    input_count = 0

    def counted(frames):
        nonlocal input_count
        for frame in frames:
            input_count += 1
            yield frame

    with VideoWriter(args.output, fps * args.multi, audio_from=args.input) as writer:
        for frame in interpolate_stream(model, counted(iter_frames(args.input)),
                                        args.multi, args.scale):
            writer.write(frame)
    print(f"{args.input} ({input_count}f @{fps:.2f}) -> {args.output} "
          f"({writer.frames_written}f @{fps * args.multi:.2f}, audio passthrough)")


if __name__ == "__main__":
    cli_main()
