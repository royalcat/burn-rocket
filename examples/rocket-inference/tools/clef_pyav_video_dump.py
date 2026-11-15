"""Dump the release's PyAV `load_video` frames for clips, to gate the ffmpeg-CLI decode path
(`clef::vision::tests::scaled_frames_match_pyav_on_clips_above_the_frame_cap`, phase 8d).

    python3 -m venv /tmp/opencode/clef/venv-av && /tmp/opencode/clef/venv-av/bin/pip install av numpy
    # clips above the 262144-pixel frame cap (swscale scaling), one with a soundtrack, one NTSC rate:
    ffmpeg -y -f lavfi -i testsrc2=size=1280x720:rate=24:duration=3 -f lavfi -i sine=frequency=440:duration=3 \
        -pix_fmt yuv420p -c:v libx264 -c:a aac -shortest /tmp/opencode/clef/vid8d/hd720.mp4
    ffmpeg -y -f lavfi -i testsrc2=size=640x480:rate=30000/1001:duration=3 -pix_fmt yuv420p -c:v libx264 -bf 2 \
        /tmp/opencode/clef/vid8d/ntsc480.mp4
    /tmp/opencode/clef/venv-av/bin/python -I tools/clef_pyav_video_dump.py /tmp/opencode/clef/vid8d hd720 ntsc480

Writes `<name>.u8` (raw uint8 `[F,H,W,3]`) and `<name>.json` (`shape`, `track`) next to each
`<name>.mp4`. Only `av` and `numpy` are needed: `torch` (imported at module level by the release
file, used only by the model code) is stubbed.
"""

import importlib.util
import json
import pathlib
import sys
import types

RELEASE = pathlib.Path(__file__).resolve().parent / "clef_joint_schema_model.py"


class Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, f=None, *a, **k):
        return f if callable(f) else self


class Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Dummy


for m in ("torch", "torch.nn", "torch.nn.functional"):
    sys.modules[m] = Stub(m)
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
spec = importlib.util.spec_from_file_location("jm", RELEASE)
jm = importlib.util.module_from_spec(spec)
sys.modules["jm"] = jm
spec.loader.exec_module(jm)

out = pathlib.Path(sys.argv[1])
for name in sys.argv[2:]:
    frames, track = jm.load_video(str(out / f"{name}.mp4"))
    (out / f"{name}.u8").write_bytes(frames.tobytes())
    (out / f"{name}.json").write_text(json.dumps({"shape": list(frames.shape), "track": track is not None}))
    print(name, frames.shape, "track" if track is not None else "silent")
