#!/usr/bin/env python3
"""
Side-by-side "before vs after" plot for the anisotropic remeshing demo.

Builds a random Delaunay mesh, runs anisotropic remeshing, and saves:
 - a "before" image of the initial mesh
 - an "after" image of the remeshed mesh
 - a combined side-by-side image for quick visual comparison

Usage examples:
    python3 demos/adapt_before_after.py --npts 120 --iters 4 \
        --out demos/adapt_pair.png --seed 5
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import numpy as np

from sofia.core.logging_utils import configure_logging, get_logger
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.anisotropic_remesh import anisotropic_local_remesh
from sofia.core.visualization import plot_mesh

# Headless-friendly backend
import matplotlib as _mpl
if not os.environ.get('MPLBACKEND'):
    try:
        _mpl.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt

log = get_logger('sofia.demo.adapt_pair')


def simple_anisotropic_metric(x: np.ndarray) -> np.ndarray:
    """Same example metric as in adapt_scenario.py (kept local for CLI use).

    Returns a 2x2 SPD matrix.
    """
    import math
    s = 0.5 * (1.0 + math.sin(2.0 * math.pi * float(x[0])))
    lam1 = 0.2 + 1.8 * s
    lam2 = 0.5
    theta = 0.3 * float(x[1])
    c = math.cos(theta)
    sn = math.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    L = np.diag([lam1, lam2])
    return R @ L @ R.T


def triplot_editor(ax: plt.Axes, editor: PatchBasedMeshEditor, title: str = ""):
    """Minimal inlined plotting of an editor's current mesh on the given axes.

    Filters out tombstoned triangles if any are present.
    """
    pts = editor.points
    tris = editor.triangles
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("editor.triangles must be (M,3)")
    mask = (tris >= 0).all(axis=1)
    tris_valid = tris[mask]
    ax.triplot(pts[:, 0], pts[:, 1], tris_valid, lw=0.6, color='k')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Before/After anisotropic remeshing comparison')
    parser.add_argument('--npts', type=int, default=60, help='Number of random points for base mesh')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument('--iters', type=int, default=3, help='Anisotropic remesh iterations')
    parser.add_argument('--out', type=str, default='adapt_pair.png', help='Output side-by-side image path')
    parser.add_argument('--alpha', type=float, default=1.5, help='alpha_split metric threshold')
    parser.add_argument('--beta', type=float, default=0.5, help='beta_collapse metric threshold')
    parser.add_argument('--tol', type=float, default=0.05, help='convergence tolerance')
    parser.add_argument('--save-individual', action='store_true', help='Also save before/after individual images')
    parser.add_argument('--show', action='store_true', help='Show interactively')
    parser.add_argument('--no-global-smoothing', action='store_true', help='Disable step 6 global metric-space smoothing')
    parser.add_argument('--no-cleanup', action='store_true', help='Disable step 7 cleanup & validation')
    args = parser.parse_args(argv)

    configure_logging(level=logging.INFO)

    # Build initial mesh
    pts, tris = build_random_delaunay(npts=args.npts, seed=args.seed)
    editor0 = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # Clone editor for processing (to keep the original intact for plotting)
    editor = PatchBasedMeshEditor(editor0.points.copy(), editor0.triangles.copy())

    log.info('Running anisotropic remeshing (npts=%d, iters=%d)', args.npts, args.iters)
    editor, info = anisotropic_local_remesh(editor, simple_anisotropic_metric,
                                            alpha_split=args.alpha,
                                            beta_collapse=args.beta,
                                            tol=args.tol,
                                            max_iter=args.iters,
                                            verbose=True,
                                            do_global_smoothing=not args.no_global_smoothing,
                                            do_cleanup=not args.no_cleanup)
    log.info('Done: %s', info)

    # Optional individual images
    if args.save_individual:
        before_path = os.path.splitext(args.out)[0] + '_before.png'
        after_path = os.path.splitext(args.out)[0] + '_after.png'
        # Ensure output directory exists
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        try:
            plot_mesh(editor0, outname=before_path)
            plot_mesh(editor, outname=after_path)
            log.info('Saved individual images: %s, %s', before_path, after_path)
        except Exception as e:
            log.warning('plot_mesh failed for individual images (%s); falling back to triplot', e)
            # Fall back to simple plots if helper API differs
            fig_i, axs_i = plt.subplots(1, 2, figsize=(10, 5))
            triplot_editor(axs_i[0], editor0, title='Before')
            triplot_editor(axs_i[1], editor, title='After')
            fig_i.tight_layout()
            fig_i.savefig(os.path.splitext(args.out)[0] + '_before_after.png', dpi=160)

    # Combined side-by-side figure
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    triplot_editor(axs[0], editor0, title='Before')
    triplot_editor(axs[1], editor, title='After')
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    log.info('Saved side-by-side image to %s', args.out)
    if args.show:
        plt.show()


if __name__ == '__main__': 
    main()
