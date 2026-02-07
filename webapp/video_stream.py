# webapp/video_stream.py
import time
from typing import Callable, Iterator, Optional


def mjpeg_generator(get_frame_bytes: Callable[[], Optional[bytes]], fps: int = 30) -> Iterator[bytes]:
    """
    Generic MJPEG stream generator.
    Browser can render it in an <img src="/video"> tag.
    """
    boundary = b"--frame"
    frame_delay = 1.0 / max(1, fps)

    while True:
        frame = get_frame_bytes()
        if frame is None:
            time.sleep(0.05)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8")
        yield frame + b"\r\n"
        time.sleep(frame_delay)
