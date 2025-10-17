#!/usr/bin/env python3
"""
Adaptation demo: build a mesh and run anisotropic remeshing guided by a simple
spatially-varying metric field. Inspired by `demos/generate_scenario.py` but
focused on demonstrating `anisotropic_local_remesh`.

Usage examples:
    python3 demos/adapt_scenario.py --npts 80 --iters 4 --out adapt_demo.png

"""
from __future__ import annotations

import argparse
import math
import os
import logging
from typing import Tuple

import numpy as np

from sofia.sofia.logging_utils import configure_logging, get_logger
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.visualization import plot_mesh
from sofia.sofia.anisotropic_remesh import anisotropic_local_remesh

# matplotlib headless backend if needed
import matplotlib as _mpl
if not os.environ.get('MPLBACKEND'):
    try:
        _mpl.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt

log = get_logger('sofia.demo.adapt')


def simple_anisotropic_metric(x: np.ndarray) -> np.ndarray:
    """Example anisotropic metric field.

    This metric stretches in the local x-direction depending on the x-coordinate.
    It returns a 2x2 SPD matrix.
    """
    # smooth modulation in [-1,1]
    s = 0.5 * (1.0 + math.sin(2.0 * math.pi * float(x[0])))
    # eigenvalues: small along one direction, large along the other
    lam1 = 0.2 + 1.8 * s   # varies between 0.2 and 2.0
    lam2 = 0.5             # relatively small other direction
    # simple rotation that slowly varies with y
    theta = 0.3 * float(x[1])
    c = math.cos(theta); sn = math.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    L = np.diag([lam1, lam2])
    return R @ L @ R.T


def main(argv=None):
    parser = argparse.ArgumentParser(description='Anisotropic remesh demo')
    parser.add_argument('--npts', type=int, default=60, help='Number of random points for base mesh')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument('--iters', type=int, default=3, help='Anisotropic remesh iterations')
    parser.add_argument('--out', type=str, default='adapt_demo.png', help='Output image path')
    parser.add_argument('--alpha', type=float, default=1.5, help='alpha_split metric threshold')
    parser.add_argument('--beta', type=float, default=0.5, help='beta_collapse metric threshold')
    parser.add_argument('--tol', type=float, default=0.05, help='convergence tolerance')
    parser.add_argument('--show', action='store_true', help='Show the plot interactively (requires display)')
    args = parser.parse_args(argv)

    configure_logging(level=logging.INFO)

    # Build a random Delaunay mesh in a unit square
    pts, tris = build_random_delaunay(npts=args.npts, seed=args.seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    log.info('Starting anisotropic remeshing: npts=%d, iters=%d', args.npts, args.iters)

    # Run anisotropic remesh
    editor, info = anisotropic_local_remesh(editor, simple_anisotropic_metric,
                                            alpha_split=args.alpha,
                                            beta_collapse=args.beta,
                                            tol=args.tol,
                                            max_iter=args.iters,
                                            verbose=True)

    log.info('Remesh finished: %s', info)

    # Plot result
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mesh(editor.points, editor.triangles, ax=ax, show_vertices=False)
    ax.set_aspect('equal')
    ax.set_title('Anisotropic remesh result')
    out_path = args.out
    fig.savefig(out_path, dpi=150)
    log.info('Saved image to %s', out_path)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
