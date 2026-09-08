"""The ffmpeg export block, extracted from the tail of every example script."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import List, Optional

__all__ = ['encodeFrames']

# The frames are fed to ffmpeg through a temp directory of symlinks named
# `f%06d.png` in true numeric order (see `_orderedFrameDir`), NOT via
# `-pattern_type glob -i 'frame_*.png'`. glob ordering is lexicographic, and
# `cases/plotting.py` names frames `frame_{step:05d}.png` -- a 5-wide pad that
# overflows the moment a run exceeds 100k steps (a high-resolution delta-SPH
# case easily does: H/dx=320 is ~156k steps). Once that happens
# `frame_100100.png` sorts before `frame_99840.png`, so every frame past step
# 100k lands in a block near the *start* of the video and the back half of the
# simulation plays out of order. A fixed `-i f%06d.png` sequence sidesteps
# filename width entirely.
_MP4 = ("ffmpeg -y -loglevel error -hide_banner -framerate {framerate} "
        "-i f%06d.png -c:v libx264 -pix_fmt yuv420p -b:v {bitrate} {out}")
_PALETTE = ('ffmpeg -y -loglevel error -hide_banner -i {mp4} '
            '-vf "fps={fps},scale={width}:-1:flags=lanczos,palettegen" palette.png')
_GIF = ('ffmpeg -y -loglevel error -hide_banner -i {mp4} -i palette.png -filter_complex '
        '"fps={gifFps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse" {gif}')

_FRAME_RE = re.compile(r'frame_(\d+)\.png$')


def _frameNumber(name: str) -> Optional[int]:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def _orderedFrames(imagePath: str) -> List[str]:
    """`frame_*.png` basenames in true numeric (step) order."""
    numbered = []
    for f in os.listdir(imagePath):
        n = _frameNumber(f)
        if n is not None:
            numbered.append((n, f))
    numbered.sort()
    return [f for _n, f in numbered]


def encodeFrames(imagePath: str,
                 destination: Optional[str] = None,
                 *,
                 framerate: int = 50,
                 fps: int = 50,
                 gifFps: int = 25,
                 width: int = 540,
                 bitrate: str = '10M',
                 pattern: str = 'frame_*.png',
                 gif: bool = True) -> Optional[str]:
    """Encode ``imagePath/frame_*.png`` to ``output.mp4`` (and ``out.gif``).

    Frames are ordered by their embedded step number, not by filename sort, so
    a run that crosses 100k steps (making ``frame_NNNNN`` widen to six digits)
    still encodes in chronological order.

    Returns the path of the mp4, or ``None`` if there was nothing to encode or
    ffmpeg is not installed -- a missing encoder should not fail a simulation
    that has already produced its frames.
    """
    if shutil.which('ffmpeg') is None:
        print('ffmpeg not found on PATH; skipping video export.')
        return None

    frames = _orderedFrames(imagePath)
    if not frames:
        return None

    mp4 = os.path.join(imagePath, 'output.mp4')
    seqDir = tempfile.mkdtemp(prefix='frames_', dir=imagePath)
    try:
        for i, name in enumerate(frames):
            # Absolute target: the symlink lives one level below `imagePath`
            # (in `seqDir`), so a relative target would resolve against the
            # wrong directory and ffmpeg would see no files.
            os.symlink(os.path.abspath(os.path.join(imagePath, name)),
                       os.path.join(seqDir, f'f{i:06d}.png'))
        subprocess.run(shlex.split(_MP4.format(framerate=framerate,
                                               bitrate=bitrate, out='output.mp4')),
                       check=True, cwd=seqDir)
        os.replace(os.path.join(seqDir, 'output.mp4'), mp4)
    finally:
        shutil.rmtree(seqDir, ignore_errors=True)

    if gif:
        subprocess.run(shlex.split(_PALETTE.format(mp4='output.mp4', fps=fps, width=width)),
                       check=True, cwd=imagePath)
        subprocess.run(shlex.split(_GIF.format(mp4='output.mp4', gifFps=gifFps, width=width,
                                               gif='out.gif')),
                       check=True, cwd=imagePath)

    if destination and os.path.abspath(destination) != os.path.abspath(imagePath):
        os.makedirs(destination, exist_ok=True)
        shutil.copy(mp4, os.path.join(destination, 'output.mp4'))
        if gif:
            shutil.copy(os.path.join(imagePath, 'out.gif'), os.path.join(destination, 'out.gif'))
        return os.path.join(destination, 'output.mp4')

    return mp4
