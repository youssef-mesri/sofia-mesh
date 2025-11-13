"""
Simple analyzer for per-patch logs produced by mesh_editor_demo.py.

Usage:
  python patch_log_analyze.py /path/to/patch_log.csv

Produces a human-readable summary and a small CSV summary `patch_log_summary.csv` in the same folder.
"""
import sys
import csv
from collections import defaultdict
import os


def analyze(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    total = len(rows)
    by_op = defaultdict(lambda: {'count':0,'ok':0,'fail':0,'dglobal':0.0})
    # compute numeric deltas
    for row in rows:
        op = row.get('op_attempted') or row.get('op') or ''
        res = row.get('result','')
        try:
            g_before = float(row.get('global_min_before') or 0.0)
            g_after = float(row.get('global_min_after') or 0.0)
        except Exception:
            g_before = 0.0; g_after = 0.0
        d = g_after - g_before
        by_op[op]['count'] += 1
        if res.startswith('ok') or res == 'ok':
            by_op[op]['ok'] += 1
        else:
            by_op[op]['fail'] += 1
        by_op[op]['dglobal'] += d

    print(f"Analyzed {total} patch attempts from {path}\n")
    print(f"Overall success rates by operation:")
    print("op, attempts, ok, fail, ok_rate, avg_delta_global_min")
    summary_rows = []
    for op, v in by_op.items():
        attempts = v['count']
        ok = v['ok']
        fail = v['fail']
        ok_rate = ok/attempts if attempts else 0.0
        avg_d = v['dglobal']/attempts if attempts else 0.0
        print(f"{op}, {attempts}, {ok}, {fail}, {ok_rate:.3f}, {avg_d:.6f}")
        summary_rows.append({'op':op,'attempts':attempts,'ok':ok,'fail':fail,'ok_rate':f"{ok_rate:.3f}", 'avg_dglobal':f"{avg_d:.6f}"})

    # write small CSV summary
    out = os.path.join(os.path.dirname(path), 'patch_log_summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['op','attempts','ok','fail','ok_rate','avg_dglobal'])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\nWrote summary to {out}")


if __name__ == '__main__': 
    if len(sys.argv) < 2:
        print('Usage: python patch_log_analyze.py /path/to/patch_log.csv')
        sys.exit(1)
    analyze(sys.argv[1])
