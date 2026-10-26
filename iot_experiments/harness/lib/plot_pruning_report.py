"""
CN: 2026-10-15 服务器批次的报告图（pruning_conservative, cupdlpx GPU master, 种入的 Monge 实例）。
EN: Report figure for the 2026-10-15 server batch (pruning_conservative, cupdlpx GPU master, planted
    Monge instances). Reads a hand-transcribed table extracted from the queue logs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (family, seed, n_eff, m_eff, gap50, noprune_it, default_conv, default_it)
DATA = {
    "tolerance, tau=1e-12": [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,12,False,100)],
    "tolerance, tau=1e-6":  [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,12,False,100)],
    "tolerance, tau=1e-3":  [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,12,False,100)],
    "tolerance, tau=1e-1":  [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,12,False,100)],
    "hysteresis, tau=1e-6": [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,11,False,100)],
    "hysteresis, tau=1e-3": [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,12,False,100)],
    "hysteresis, tau=1e-1": [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,True,9),
                             ("diagonal",0,3200,1926,0.0131,10,False,100),("diagonal",1,3200,1867,0.0137,12,True,17)],
    "hyst+reg eps=1e-6":    [("quadratic",0,3200,71,0.410,8,False,100),("quadratic",1,3200,62,0.384,8,False,100),
                             ("diagonal",0,3200,1926,0.0131,12,False,100),("diagonal",1,3200,1867,0.0137,12,False,100)],
    "hyst+reg eps=1e-4":    [("quadratic",0,3200,71,0.410,8,True,13),("quadratic",1,3200,62,0.384,8,True,9),
                             ("diagonal",0,3200,1926,0.0131,10,True,30),("diagonal",1,3200,1867,0.0137,12,True,24)],
}
SCALE = [  # (n, noprune_it, noprune_conv, hyst_tau1e-3_it, hyst_tau1e-3_conv)
    (6400, 9, True, 100, False),
    (12800, 10, True, 100, False),
]
JOB_WALL = [("tol 1e-12",1753),("tol 1e-6",1540),("tol 1e-3",2547),("tol 1e-1",1769),
            ("hyst 1e-6",6400),("hyst 1e-3",1523),("hyst 1e-1",759),
            ("hyst+reg 1e-6",7187),("hyst+reg 1e-4",858),
            ("scale noprune",2219),("scale hyst 1e-3",4539)]

labels = [k for k in DATA]
ncfg = len(next(iter(DATA.values())))
frac_conv = [sum(1 for r in DATA[k] if r[6]) / len(DATA[k]) for k in labels]
noprune_it = [np.mean([r[5] for r in DATA[k]]) for k in labels]
default_it = [np.mean([r[7] for r in DATA[k]]) for k in labels]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

# --- (a) convergence rate per strategy ---
ax = axes[0]
colors = ["#c44e52" if f == 0 else ("#dd8452" if f < 1 else "#55a868") for f in frac_conv]
ax.barh(labels, frac_conv, color=colors)
ax.set_xlim(0, 1.05)
ax.set_xlabel("fraction of instances that CONVERGED\n(4 instances per strategy unless noted)")
ax.set_title("(a) Does pruning converge?\ncupdlpx GPU master, planted Monge, n_eff=3200")
for i, f in enumerate(frac_conv):
    ax.text(max(f, 0.02) + 0.02, i, f"{f:.0%}", va="center", fontsize=9)
ax.axvline(1.0, color="k", lw=0.8, ls=":")

# --- (b) iterations: prune=none vs the pruning strategy ---
ax = axes[1]
y = np.arange(len(labels))
ax.barh(y - 0.2, noprune_it, height=0.4, color="#4c72b0", label="prune=none")
ax.barh(y + 0.2, default_it, height=0.4, color="#c44e52", label="pruning strategy")
ax.set_yticks(y); ax.set_yticklabels([])
ax.axvline(100, color="k", lw=0.8, ls="--")
ax.text(100, len(labels) - 0.4, " it cap (100)", fontsize=8)
ax.set_xlabel("mean iterations (cap 100)")
ax.set_title("(b) Iterations\n(all pruning arms hit the cap except tau=1e-1 / reg eps=1e-4)")
ax.legend(loc="lower right", fontsize=9)

# --- (c) job wall-clock ---
ax = axes[2]
names = [w[0] for w in JOB_WALL]; walls = [w[1] for w in JOB_WALL]
ax.barh(names, walls, color="#8172b3")
ax.set_xlabel("job wall-clock (s)")
ax.set_title("(c) Cost of one job\n(CPU-bound; GPU ~3% busy)")
for i, w in enumerate(walls):
    ax.text(w + 80, i, f"{w/60:.0f}m", va="center", fontsize=9)

fig.suptitle(
    "Conservative pruning on the cupdlpx GPU master — planted Monge instances (2026-10-15, A100)\n"
    "n_eff=3200, d=8, quadratic/diagonal, seeds 0-1, g=0.  cert exact (0.0) in every instance.",
    fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = "iot_experiments/core/out/planting/pruning_report.png"
import pathlib
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out)
