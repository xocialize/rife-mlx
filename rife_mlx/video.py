"""Incremental video decode/encode with packet-copied audio passthrough."""

from __future__ import annotations

import fractions
import heapq
import os
import tempfile
from collections.abc import Iterable, Iterator

import numpy as np


def probe_fps(path: str) -> float:
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        return float(stream.average_rate) if stream.average_rate else 30.0


def iter_frames(path: str) -> Iterator[np.ndarray]:
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        for frame in container.decode(container.streams.video[0]):
            yield frame.to_ndarray(format="rgb24")


def read_frames(path: str):
    """Compatibility helper returning all RGB frames and the source frame rate."""
    return list(iter_frames(path)), probe_fps(path)


def _temp_path(path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    suffix = os.path.splitext(path)[1] or ".mp4"
    fd, name = tempfile.mkstemp(prefix=".rife-mlx-", suffix=suffix, dir=directory)
    os.close(fd)
    return name


def _remux(video_path: str, audio_from: str, output_path: str) -> None:
    """Merge two packet streams in timestamp order, retaining one packet per stream."""
    import av

    with av.open(video_path) as video, av.open(audio_from) as source:
        if not video.streams.video:
            raise ValueError("encoded output has no video stream")
        video_in = video.streams.video[0]
        audio_in = source.streams.audio[0] if source.streams.audio else None
        if audio_in is None:
            os.replace(video_path, output_path)
            return

        merged_path = _temp_path(output_path)
        try:
            with av.open(merged_path, mode="w") as output:
                video_out = output.add_stream_from_template(video_in)
                audio_out = output.add_stream_from_template(audio_in)

                def packets(container, in_stream, out_stream, order):
                    for packet in container.demux(in_stream):
                        if packet.dts is None:
                            continue
                        timestamp = packet.dts * (packet.time_base or in_stream.time_base)
                        packet.stream = out_stream
                        yield timestamp, order, packet

                streams = [
                    iter(packets(video, video_in, video_out, 0)),
                    iter(packets(source, audio_in, audio_out, 1)),
                ]
                pending = []
                for index, packets_iter in enumerate(streams):
                    packet = next(packets_iter, None)
                    if packet is not None:
                        heapq.heappush(pending, (packet[0], packet[1], index, packet[2]))

                while pending:
                    _timestamp, _order, index, packet = heapq.heappop(pending)
                    output.mux(packet)
                    next_packet = next(streams[index], None)
                    if next_packet is not None:
                        heapq.heappush(pending, (next_packet[0], next_packet[1], index,
                                                 next_packet[2]))
            os.replace(merged_path, output_path)
        except BaseException:
            try:
                os.unlink(merged_path)
            except FileNotFoundError:
                pass
            raise


class VideoWriter:
    """Write frames incrementally and publish the completed file atomically."""

    def __init__(self, path: str, fps: float, audio_from: str | None = None) -> None:
        import av

        self._av = av
        self.path = os.fspath(path)
        self.fps = fps
        self.audio_from = audio_from
        self.frames_written = 0
        self._temp = _temp_path(self.path)
        self._out = av.open(self._temp, mode="w")
        self._video = None

    def write(self, frame: np.ndarray) -> None:
        if self._out is None:
            raise RuntimeError("writer is closed")
        if self._video is None:
            height, width = frame.shape[:2]
            self._video = self._out.add_stream(
                "h264", rate=fractions.Fraction(self.fps).limit_denominator(1000000))
            self._video.width = width
            self._video.height = height
            self._video.pix_fmt = "yuv420p"

        video_frame = self._av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame), format="rgb24")
        for packet in self._video.encode(video_frame):
            self._out.mux(packet)
        self.frames_written += 1

    def close(self) -> None:
        if self._out is None:
            return
        try:
            if self._video is None:
                raise ValueError("cannot write a video with no frames")
            for packet in self._video.encode():
                self._out.mux(packet)
            self._out.close()
            self._out = None
            if self.audio_from is not None:
                _remux(self._temp, self.audio_from, self.path)
                if os.path.exists(self._temp):
                    os.unlink(self._temp)
            else:
                os.replace(self._temp, self.path)
            self._temp = ""
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None
        if self._temp:
            try:
                os.unlink(self._temp)
            except FileNotFoundError:
                pass
            self._temp = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_video(path: str, frames: Iterable[np.ndarray], fps: float,
                audio_from: str | None = None) -> None:
    """Compatibility helper writing an iterable without collecting it."""
    with VideoWriter(path, fps, audio_from) as writer:
        for frame in frames:
            writer.write(frame)
