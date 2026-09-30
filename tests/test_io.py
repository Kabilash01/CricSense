

def test_reader_skips_corrupt_frames_and_keeps_ids_on_time(tmp_path):
    import cv2
    import numpy as np

    from cricsense.io import VideoReader

    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()

    class Flaky:  # frames 5 and 6 fail to decode, as corrupt packets in a long broadcast do
        def __init__(self, cap):
            self.cap, self.n = cap, 0

        def _next(self, read):
            i, self.n = self.n, self.n + 1
            ok, img = self.cap.read() if read else (self.cap.grab(), None)
            return (False, None) if i in (5, 6) else (ok, img)

        def read(self):
            return self._next(True)

        def grab(self):
            return self._next(False)[0]

        def __getattr__(self, name):
            return getattr(self.cap, name)

    with VideoReader(path) as r:
        r._cap = Flaky(r._cap)
        ids = [f.frame_id for f in r]
    assert ids == [i for i in range(20) if i not in (5, 6)]  # past the bad frames, ids still on time
