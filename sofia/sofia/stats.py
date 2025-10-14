"""Operation statistics data structures and presentation utilities.

Centralizes the OpStats dataclass and helpers that were previously
implemented inside `mesh_modifier2.PatchBasedMeshEditor` so that the
editor file can focus on orchestration logic only.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OpStats:
    attempts: int = 0
    success: int = 0
    fail: int = 0
    quality_rejects: int = 0
    simulation_rejects: int = 0
    fallback_used: int = 0
    # Remove-node specific extras (safe no-ops for others)
    star_success: int = 0
    simplify_attempted: int = 0
    simplify_helped: int = 0
    # Pocket fill strategy breakdown
    pocket_quad_attempts: int = 0
    pocket_quad_success: int = 0
    pocket_steiner_attempts: int = 0
    pocket_steiner_success: int = 0
    pocket_earclip_attempts: int = 0
    pocket_earclip_success: int = 0
    # Timing (seconds)
    time_total: float = 0.0
    time_max: float = 0.0
    time_min: float = 0.0  # 0 means uninitialized

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - simple mapping
        return {
            'attempts': self.attempts,
            'success': self.success,
            'fail': self.fail,
            'quality_rejects': self.quality_rejects,
            'simulation_rejects': self.simulation_rejects,
            'fallback_used': self.fallback_used,
            'star_success': self.star_success,
            'simplify_attempted': self.simplify_attempted,
            'simplify_helped': self.simplify_helped,
            'success_rate': (self.success / self.attempts) if self.attempts else 0.0,
            'fallback_rate': (self.fallback_used / self.attempts) if self.attempts else 0.0,
            'quality_reject_rate': (self.quality_rejects / self.attempts) if self.attempts else 0.0,
            'simulation_reject_rate': (self.simulation_rejects / self.attempts) if self.attempts else 0.0,
            'simplify_help_rate': (self.simplify_helped / self.simplify_attempted) if self.simplify_attempted else 0.0,
            'pocket_quad_attempts': self.pocket_quad_attempts,
            'pocket_quad_success': self.pocket_quad_success,
            'pocket_steiner_attempts': self.pocket_steiner_attempts,
            'pocket_steiner_success': self.pocket_steiner_success,
            'pocket_earclip_attempts': self.pocket_earclip_attempts,
            'pocket_earclip_success': self.pocket_earclip_success,
            'pocket_quad_success_rate': (self.pocket_quad_success / self.pocket_quad_attempts) if self.pocket_quad_attempts else 0.0,
            'pocket_steiner_success_rate': (self.pocket_steiner_success / self.pocket_steiner_attempts) if self.pocket_steiner_attempts else 0.0,
            'pocket_earclip_success_rate': (self.pocket_earclip_success / self.pocket_earclip_attempts) if self.pocket_earclip_attempts else 0.0,
            'time_total': self.time_total,
            'time_max': self.time_max,
            'time_min': (self.time_min if self.time_min != 0.0 else 0.0),
            'time_avg': (self.time_total / self.attempts) if self.attempts else 0.0,
        }

def format_stats_table(stats_dict) -> str:
    """Return a human readable multi-line table summarizing op stats."""
    if not stats_dict:
        return "<no stats>"
    header = ["op", "attempts", "succ", "fail", "qualRej", "simRej", "succ%", "avg_ms", "min_ms", "max_ms"]
    rows = []
    for op in sorted(stats_dict.keys()):
        s = stats_dict[op]
        attempts = s['attempts']; succ = s['success']; fail = s['fail']
        qual = s['quality_rejects']; simr = s['simulation_rejects']
        succ_pct = (succ / attempts * 100.0) if attempts else 0.0
        avg_ms = s['time_avg'] * 1000.0; min_ms = s['time_min'] * 1000.0; max_ms = s['time_max'] * 1000.0
        rows.append([
            op, str(attempts), str(succ), str(fail), str(qual), str(simr),
            f"{succ_pct:6.2f}", f"{avg_ms:8.3f}", f"{min_ms:8.3f}", f"{max_ms:8.3f}"
        ])
    col_w = [len(h) for h in header]
    for r in rows:
        for i,v in enumerate(r):
            if len(v) > col_w[i]: col_w[i] = len(v)
    def fmt(r):
        return " ".join(r[i].rjust(col_w[i]) for i in range(len(r)))
    lines = [fmt(header), "-" * (sum(col_w) + len(col_w) - 1)] + [fmt(r) for r in rows]
    return "\n".join(lines)

def print_stats(stats_dict, file=None, pretty=True):  # pragma: no cover - formatting wrapper
    import sys
    out = file or sys.stdout
    if not pretty:
        print(stats_dict, file=out)
        return
    print(format_stats_table(stats_dict), file=out)
    # Optional pocket strategy breakdown
    if 'fill_pocket' in stats_dict:
        s = stats_dict['fill_pocket']
        if any(s.get(k,0) for k in ['pocket_quad_attempts','pocket_steiner_attempts','pocket_earclip_attempts']):
            strat_header = ["strategy","attempts","succ","succ%"]
            entries = []
            for label, a_key, s_key in [
                ("quad", 'pocket_quad_attempts','pocket_quad_success'),
                ("steiner", 'pocket_steiner_attempts','pocket_steiner_success'),
                ("earclip", 'pocket_earclip_attempts','pocket_earclip_success')]:
                a = s.get(a_key,0); su = s.get(s_key,0)
                pct = (su / a * 100.0) if a else 0.0
                entries.append([label, str(a), str(su), f"{pct:6.2f}"])
            cw = [len(h) for h in strat_header]
            for r in entries:
                for i,v in enumerate(r):
                    cw[i] = max(cw[i], len(v))
            def fmt_s(r): return " ".join(r[i].rjust(cw[i]) for i in range(len(r)))
            print("\nPocket strategies:", file=out)
            print(fmt_s(strat_header), file=out)
            print("-" * (sum(cw) + len(cw) - 1), file=out)
            for r in entries:
                print(fmt_s(r), file=out)

__all__ = ["OpStats", "print_stats", "format_stats_table"]

