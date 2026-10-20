"""Video publication and temporary-file cleanup, without model weights."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import av

from rife_mlx.video import VideoWriter, iter_frames, write_video


class VideoWriterTests(unittest.TestCase):
    def test_remux_removes_temporary_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "output.mp4"
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            with av.open(str(source), "w") as container:
                video = container.add_stream("h264", rate=30)
                video.width = video.height = 16
                video.pix_fmt = "yuv420p"
                audio = container.add_stream("aac", rate=48000)
                audio.layout = "mono"
                for packet in video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
                for packet in video.encode():
                    container.mux(packet)
                samples = av.AudioFrame.from_ndarray(
                    np.zeros((1, 1024), dtype=np.float32), format="fltp", layout="mono")
                samples.sample_rate = 48000
                samples.pts = 0
                for packet in audio.encode(samples):
                    container.mux(packet)
                for packet in audio.encode():
                    container.mux(packet)

            def audio_packets(path):
                with av.open(str(path)) as container:
                    return [bytes(packet) for packet in container.demux(container.streams.audio[0])
                            if packet.dts is not None]

            write_video(output, iter([frame, frame]), 60, audio_from=source)
            self.assertEqual(len(list(iter_frames(output))), 2)
            self.assertTrue(audio_packets(source))
            self.assertEqual(audio_packets(source), audio_packets(output))
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["output.mp4", "source.mp4"])

    def test_no_audio_remux_moves_temporary_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "output.mp4"
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            write_video(source, [frame], 30)
            write_video(output, [frame], 30, audio_from=source)
            self.assertEqual(len(list(iter_frames(output))), 1)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["output.mp4", "source.mp4"])

    def test_remux_failure_preserves_output_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            output.write_bytes(b"existing output")
            with patch("rife_mlx.video._remux", side_effect=RuntimeError("remux failed")):
                with self.assertRaisesRegex(RuntimeError, "remux failed"):
                    with VideoWriter(output, 30, audio_from="unused.mp4") as writer:
                        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
            self.assertEqual(output.read_bytes(), b"existing output")
            self.assertEqual(list(Path(directory).iterdir()), [output])
