"""Unified configuration objects for Sofia remeshing drivers.

This module centralizes tunable parameters that were previously scattered
across global variables (in `remesh_driver`) and the `PatchDriverConfig`
dataclass inside `patch_driver`. A single `RemeshConfig` can be passed
to greedy or patch drivers; a light adapter exposes the legacy
`PatchDriverConfig` shape for backwards compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from .constants import EPS_AREA

@dataclass
class GreedyConfig:
    max_vertex_passes: int = 1
    max_edge_passes: int = 1
    strict: bool = False
    reject_crossings: bool = False
    reject_new_loops: bool = False
    force_pocket_fill: bool = False
    verbose: bool = False
    gif_capture: bool = False
    gif_dir: str = 'greedy_frames'
    gif_out: str = 'greedy_run.gif'
    gif_fps: int = 4
    allow_flips: bool = True
    min_triangle_area: float = EPS_AREA
    reject_min_angle_deg: Optional[float] = None
    # Driver-level amortization for strict crossing simulation (default 0 = disabled)
    strict_check_cooldown: int = 0
    strict_check_risk_threshold: int = 12
    # Compact triangles/vertices at the end of each pass (vertex and edge)
    compact_end_of_pass: bool = False

@dataclass
class PatchConfig:
    threshold: float = 20.0
    max_iterations: int = 500
    plot_every: int = 50
    patch_radius: int = 1
    top_k: int = 80
    disjoint_on: str = 'tri'
    allow_overlap: bool = False
    batch_attempts: int = 2
    min_triangle_area: float = EPS_AREA
    min_triangle_area_fraction: Optional[float] = None
    reject_min_angle_deg: Optional[float] = None
    # Guards analogous to greedy mode
    reject_new_loops: bool = True
    reject_crossings: bool = False
    auto_fill_pockets: bool = False
    autofill_min_triangle_area: Optional[float] = None
    autofill_reject_min_angle_deg: Optional[float] = None
    angle_unit: str = 'deg'
    log_dir: Optional[str] = None
    out_prefix: str = 'debug_mesh'
    use_greedy_remesh: bool = False
    greedy_vertex_passes: int = 1
    greedy_edge_passes: int = 1
    gif_capture: bool = False
    gif_dir: str = 'patch_frames'
    gif_out: str = 'patch_run.gif'
    gif_fps: int = 4

@dataclass
class RemeshConfig:
    """Unified configuration.

    Attributes
    ----------
    greedy : GreedyConfig
        Parameters for greedy remeshing.
    patch : PatchConfig
        Parameters for patch/batch remeshing.
    extras : dict
        Free-form dictionary for future extensions (logging hooks, callbacks).
    """
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_patch_config(cls, patch_cfg: PatchConfig, *, greedy_overrides: Dict[str, Any]=None) -> 'RemeshConfig':
        g = GreedyConfig()
        if patch_cfg.use_greedy_remesh:
            g.max_vertex_passes = patch_cfg.greedy_vertex_passes
            g.max_edge_passes = patch_cfg.greedy_edge_passes
            g.gif_capture = patch_cfg.gif_capture
            g.gif_dir = patch_cfg.gif_dir
            g.gif_out = patch_cfg.gif_out.replace('patch_', 'greedy_') if patch_cfg.gif_out else 'greedy_run.gif'
        if greedy_overrides:
            for k,v in greedy_overrides.items():
                setattr(g, k, v)
        return cls(greedy=g, patch=patch_cfg)

# Backwards compatibility alias (external tests may still import PatchDriverConfig)
PatchDriverConfig = PatchConfig

__all__ = [
    'GreedyConfig','PatchConfig','RemeshConfig','PatchDriverConfig'
]
