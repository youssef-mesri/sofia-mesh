"""Unified GIF capture helper to avoid duplicated frame logic.

Usage pattern:
    from gif_capture import GifCapture
    cap = GifCapture(enabled=True, directory='frames', outfile='run.gif', fps=4, max_frames=800, logger=logger)
    cap.save(editor, 'start', plot_fn=lambda ed, path: plot_mesh(ed, outname=path))
    ... during operations ...
    cap.save(editor, 'event_tag', plot_fn=...)
    cap.finalize()

The helper is resilient to errors (silent failures) and gracefully degrades if
`imageio` is not installed (frames still written; GIF skipped).
"""
from __future__ import annotations

import os
import logging
from typing import Callable, Optional
from .logging_utils import get_logger

try:  # optional dependency
    import imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

class GifCapture:
    def __init__(self, enabled: bool, directory: str, outfile: str, fps: int = 4, max_frames: int = 1000, logger: Optional[logging.Logger] = None):
        self.enabled = bool(enabled)
        self.directory = directory
        self.outfile = outfile
        self.fps = max(1, int(fps))
        self.max_frames = max_frames
        self.logger = logger or get_logger('sofia.gif')
        self.frames = []  # list of file paths
        if self.enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except Exception as e:  # pragma: no cover
                self.logger.debug('GifCapture: could not create directory %s (%s); disabling', self.directory, e)
                self.enabled = False

    def save(self, editor, tag: str, plot_fn: Callable[[object, str], None]):
        if not self.enabled:
            return
        if len(self.frames) >= self.max_frames:
            return
        # Sanitize tag
        safe_tag = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in str(tag))[:60]
        fname = os.path.join(self.directory, f"frame_{len(self.frames):05d}_{safe_tag}.png")
        try:
            plot_fn(editor, fname)
            self.frames.append(fname)
        except Exception as e:  # pragma: no cover
            self.logger.debug('GifCapture: frame save failed (%s)', e)

    def finalize(self):
        if not self.enabled or len(self.frames) < 2:
            return None
        if imageio is None:  # pragma: no cover
            self.logger.info('GifCapture: imageio not available; skipping GIF assembly')
            return None
        images = []
        for f in self.frames:
            try:
                images.append(imageio.v2.imread(f))
            except Exception:  # pragma: no cover
                pass
        if not images:
            return None
        out_path = os.path.join(self.directory, self.outfile)
        try:
            imageio.mimsave(out_path, images, fps=self.fps)
            self.logger.info('GifCapture: wrote %s (%d frames)', out_path, len(images))
            return out_path
        except Exception as e:  # pragma: no cover
            self.logger.debug('GifCapture: assembly failed (%s)', e)
            return None

__all__ = ['GifCapture']
