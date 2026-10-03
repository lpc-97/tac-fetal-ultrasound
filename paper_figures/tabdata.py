"""Row statistics recomputed from results/qc_collect_2026-09-18.json (the provenance collection behind the QC workbook).
Every number used in a figure comes through here: 6-direction mean per seed -> mean / population std over seeds 42/43/44."""
import math
import statistics as st

from pubstyle import load

QC = load("qc_collect_2026-09-18.json")
T975 = 4.303  # two-sided 97.5% t quantile, 2 dof (3 seeds)


def rows(table):
    """dict name -> {mean, std, ap50, views, spi, dirs (6-dir means over seeds), m6 (per-seed 6-dir means), dirkeys}"""
    out = {}
    for name, cells in QC["tables"][table].items():
        per = {}
        for k, v in cells.items():
            per.setdefault(k.split("|")[0], []).append(v)
        ok = {s: v for s, v in per.items() if len(v) == 6 and all(x.get("AP") is not None for x in v)}
        if not ok:
            continue
        m6 = {s: st.mean(x["AP"] for x in v) for s, v in ok.items()}
        m50 = {s: st.mean(x["AP50"] for x in v) for s, v in ok.items()}
        views = [x["views"] for v in ok.values() for x in v if x.get("views") is not None]
        spi = [x["s_per_img"] for v in ok.values() for x in v if x.get("s_per_img") is not None]
        dirs = [st.mean(ok[s][i]["AP"] for s in ok) for i in range(6)]
        out[name] = {"mean": st.mean(m6.values()), "std": st.pstdev(m6.values()) if len(m6) > 1 else 0.0, "n": len(m6),
                     "ap50": st.mean(m50.values()), "views": st.mean(views) if views else None, "spi": st.mean(spi) if spi else None,
                     "dirs": dirs, "m6": m6, "m50": m50, "dirkeys": [k.split("|")[1] for k in list(cells)[:6]]}
    return out


def paired(a, b):
    """paired delta a-b over seeds: (mean, 95% half-width, significant)"""
    d = [a["m6"][s] - b["m6"][s] for s in a["m6"] if s in b["m6"]]
    mu = st.mean(d); half = T975 * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return mu, half, abs(mu) > half


def val_variants(dsn, folder):
    """6-dir mean AP on val (seed 42) per variant"""
    by = {}
    for e in QC["val_sweeps"][dsn]:
        if e["folder"] == folder:
            by.setdefault(e["variant"], {})[e["dir"]] = e["AP"]
    return {v: st.mean(d.values()) for v, d in by.items() if len(d) == 6}
