"""Clean run viewer: details + key graphs only. No animation.

Usage:
  python ui/render.py --run /path/to/results/bldc/<study> --out /tmp/vroom-ui

Reads (best-effort, never crashes on missing files):
  status.json, selection.json, protocol.json, queue_status.json, stop_record.json
  */*/status.json (job status with outer_actions/selected)
  */validation/*/metrics.json or validation/*/metrics.json (mean RL RMSE curve)

Writes:
  out/index.html + out/*.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def find_jobs(root):
    jobs = []
    seen = set()
    for p in sorted(root.rglob("manifest.json")) + sorted(root.rglob("status.json")):
        d = load_json(p)
        if not isinstance(d, dict):
            continue
        if "outer_actions" not in d and "selected" not in d:
            continue
        job = p.parent.relative_to(root).as_posix()
        if job in seen:
            # prefer manifest.json (has outer_actions+selected); status.json is often summary-only
            if p.name == "status.json":
                continue
        seen.add(job)
        sel = d.get("selected") or {}
        jobs.append({
            "job": job,
            "status": d.get("status", "?"),
            "outer_actions": d.get("outer_actions"),
            "failed": d.get("failed_episodes"),
            "sel_step": sel.get("step"),
            "passed": sel.get("passed_cases"),
            "cases": sel.get("case_count"),
            "rmse": (sel.get("score") or [None, None, None])[2],
        })
    # merge selection.json rows for jobs missing manifest (e.g. interrupted runs)
    return sorted(jobs, key=lambda j: j["job"])


def rl_rmse_from_metrics(metrics):
    """Mean RL rmse across cases. Each case dict holds e.g. {'rl': {...rmse...}}."""
    vals = []
    if not isinstance(metrics, list):
        return None
    for case in metrics:
        if not isinstance(case, dict):
            continue
        entry = case.get("rl")
        if entry is None:
            for k, v in case.items():
                if k in ("case", "pi", "all_pi"):
                    continue
                if isinstance(v, dict) and "rmse_rad_s" in v:
                    entry = v
                    break
        if isinstance(entry, dict) and entry.get("rmse_rad_s") is not None:
            vals.append(entry["rmse_rad_s"])
    if not vals:
        return None
    return sum(vals) / len(vals)


def validation_curve(job_dir):
    pts = []
    for m in sorted(job_dir.glob("validation/*/metrics.json")):
        try:
            step = int(m.parent.name)
        except ValueError:
            continue
        rmse = rl_rmse_from_metrics(load_json(m))
        if rmse is not None:
            pts.append((step, rmse))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.run)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    status = load_json(root / "status.json")
    selection = load_json(root / "selection.json")
    protocol = load_json(root / "protocol.json")
    stop = load_json(root / "stop_record.json")
    jobs = find_jobs(root)

    # selection.json may be {"rows": [...]} (outer studies) — merge as fallback
    sel_rows = []
    if isinstance(selection, dict) and isinstance(selection.get("rows"), list):
        sel_rows = selection["rows"]
    elif isinstance(selection, list):
        sel_rows = selection

    plt.rcParams.update({"font.size": 10})
    figs = []

    # Fig1: selected RMSE per job
    if jobs and any(j["rmse"] is not None for j in jobs):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        labels = [j["job"] for j in jobs]
        vals = [j["rmse"] if j["rmse"] is not None else 0 for j in jobs]
        ax.bar(range(len(vals)), vals, color="#2563eb")
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Selected mean speed RMSE (rad/s)")
        ax.set_title("Selected checkpoint — mean speed RMSE")
        fig.tight_layout()
        fig.savefig(out / "selected_rmse.png", dpi=150)
        plt.close(fig)
        figs.append("selected_rmse.png")

    # Fig2: validation curves
    curves = 0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for j in jobs:
        pts = validation_curve(root / j["job"])
        if len(pts) >= 2:
            ax.plot([s for s, _ in pts], [r for _, r in pts], marker="o", ms=3, label=j["job"])
            curves += 1
    if curves:
        ax.set_xlabel("Validation step (outer actions)")
        ax.set_ylabel("Mean RL speed RMSE (rad/s)")
        ax.set_title("Validation curves")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "validation_curves.png", dpi=150)
        figs.append("validation_curves.png")
    plt.close(fig)

    # Fig3: passed cases per job
    if jobs and any(j["passed"] is not None for j in jobs):
        fig, ax = plt.subplots(figsize=(8, 3.0))
        labels = [j["job"] for j in jobs]
        passed = [j["passed"] or 0 for j in jobs]
        cases = [j["cases"] or 0 for j in jobs]
        ax.bar(range(len(passed)), passed, color="#16a34a", label="passed")
        if any(cases):
            ax.bar(range(len(cases)), [c - p for c, p in zip(cases, passed)],
                   bottom=passed, color="#e5e7eb", label="failed")
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Cases")
        ax.set_title("Development gates passed (selected checkpoint)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "passed_cases.png", dpi=150)
        plt.close(fig)
        figs.append("passed_cases.png")

    # HTML
    def row(j):
        rmse = f"{j['rmse']:.5f}" if isinstance(j["rmse"], (int, float)) else "—"
        passed = f"{j['passed']}/{j['cases']}" if j["passed"] is not None else "—"
        return (f"<tr><td>{j['job']}</td><td>{j['status']}</td>"
                f"<td>{j['outer_actions']}</td><td>{j['failed']}</td>"
                f"<td>{j['sel_step']}</td><td>{passed}</td><td>{rmse}</td></tr>")

    run_status = (status or {}).get("status", "unknown") if isinstance(status, dict) else "unknown"
    proto_note = ""
    if isinstance(protocol, dict):
        proto_note = f"protocol keys: {', '.join(list(protocol)[:8])}"
    stop_note = ""
    if isinstance(stop, dict):
        stop_note = f"<p><b>Stop:</b> {stop.get('status')} @ {stop.get('stopped_utc')} — {stop.get('reason','')}</p>"
    sel_note = ""
    if sel_rows:
        sel_note = f"<p>Selection rows: {len(sel_rows)}. Winner detail in selection.json.</p>"

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>vroom run — {root.name}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#111}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}}
th{{background:#f3f4f6}}img{{max-width:100%;border:1px solid #e5e7eb;margin:12px 0}}
.muted{{color:#555}}.card{{border:1px solid #e5e7eb;padding:12px 16px;margin:12px 0}}</style></head><body>
<h1>{root.name}</h1>
<div class=card><b>Status:</b> {run_status}<br>
<span class=muted>{root}</span><br><span class=muted>{proto_note}</span></div>
{stop_note}{sel_note}
<h2>Jobs</h2>
<table><tr><th>job</th><th>status</th><th>outer actions</th><th>failed eps</th>
<th>sel step</th><th>passed</th><th>sel RMSE rad/s</th></tr>
{''.join(row(j) for j in jobs) or '<tr><td colspan=7>No job status.json found</td></tr>'}
</table>
<h2>Graphs</h2>
{''.join(f'<h3>{f}</h3><img src="{f}">' for f in figs) or '<p class=muted>No graphs available for this run.</p>'}
<p class=muted>Sim evidence only. Development cases reused; not fresh final tests.</p>
</body></html>"""
    (out / "index.html").write_text(html)
    print(f"wrote {out/'index.html'} + {len(figs)} pngs, {len(jobs)} jobs")


if __name__ == "__main__":
    main()
