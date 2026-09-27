"""Merge 32c/96c/GMaster times and NaMaster residuals into one plot JSON."""
from __future__ import annotations

import json
from pathlib import Path


def _idx(rows, kind):
    return {(r["nside"], r["spin"]): r for r in rows.get(kind, [])}


def _get(d, *keys, default=float("nan")):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur == cur else default


def _acc(gm_row, ind_row, prec, key):
    v = _get(gm_row, prec, "accuracy", key)
    if isinstance(v, float) and v == v:
        return v
    return _get(ind_row, prec, "accuracy", key)


def main():
    root = Path(__file__).resolve().parent
    nm32 = json.loads((root / "results_nm32.json").read_text())
    nm96 = json.loads((root / "results_nm96.json").read_text())
    gm = json.loads((root / "results_gm_noprefill.json").read_text())
    acc_src = root / "results_v2_independent.json"
    acc = json.loads(acc_src.read_text()) if acc_src.exists() else {"sht": [], "pipeline": []}

    p32, p96, pg, pa = _idx(nm32, "pipeline"), _idx(nm96, "pipeline"), _idx(gm, "pipeline"), _idx(acc, "pipeline")
    hist_src = root / "results_gm_4096s2_historical.json"
    hist = json.loads(hist_src.read_text()) if hist_src.exists() else {"pipeline": []}
    ph = _idx(hist, "pipeline")
    key4096s2 = (4096, 2)
    used_hist = False
    if key4096s2 in ph:
        live = pg.get(key4096s2, {})
        live_t = _get(live, "fp32", "times", "total")
        if not (isinstance(live_t, float) and live_t == live_t):
            pg[key4096s2] = ph[key4096s2]
            used_hist = True
    s32, s96, sg, sa = _idx(nm32, "sht"), _idx(nm96, "sht"), _idx(gm, "sht"), _idx(acc, "sht")
    pipe_keys = sorted(set(p32) | set(p96) | set(pg) | set(pa))
    sht_keys = sorted(set(s32) | set(s96) | set(sg) | set(sa))

    plot = {
        "meta": {
            "gmaster": "GPU 1, GMASTER_MARCH_V2=1, XLA_PYTHON_CLIENT_PREALLOCATE=false, OMP=8",
            "namaster_32": "taskset 0-31, OMP=32",
            "namaster_96": "taskset 0-95, OMP=96",
            "times_from": [
                "results_gm_noprefill.json",
                "results_nm32.json",
                "results_nm96.json",
            ] + (["results_gm_4096s2_historical.json"] if used_hist else []),
            "errors_from": "results_v2_independent.json; pipeline nside=4096 spin=2 from GMaster+ref prefill run",
            "error_is": "one residual per (nside, spin, precision) after the timed cell, not per-repeat samples",
            "gmaster_4096_spin2_pipeline": (
                "historical: .qwen/tmp/chain_s37j.log fp32 8805 ms, report fp64 14210 ms (this campaign OOM on GPU0/GPU1)"
                if used_hist
                else "measured this campaign"
            ),
        },
        "sht": [],
        "pipeline": [],
    }

    md = [
        "# GMaster v2 vs NaMaster 32-core and 96-core (times + residuals)",
        "",
        "Times: noprefill GMaster + pinned NaMaster. Residuals: independent same-kernel run vs NaMaster.",
        "",
        "## 1. map2alm (ms) and relative $\\Delta a_{\\ell m}$",
        "",
        "| Nside | spin | DUCC 32c | DUCC 96c | GM fp32 | GM fp64 | fp32 rel $\\Delta a$ | fp64 rel $\\Delta a$ | fp32 rms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v, nd=2):
        return f"{v:.{nd}f}" if isinstance(v, float) and v == v else "—"

    def sci(v):
        return f"{v:.2e}" if isinstance(v, float) and v == v else "—"

    for key in sht_keys:
        ns, sp = key
        a32, a96, ag, aa = s32.get(key, {}), s96.get(key, {}), sg.get(key, {}), sa.get(key, {})
        row = {
            "nside": ns,
            "spin": sp,
            "ducc32_ana_ms": _get(a32, "ref", "ana_ms"),
            "ducc96_ana_ms": _get(a96, "ref", "ana_ms"),
            "ducc32_syn_ms": _get(a32, "ref", "syn_ms"),
            "ducc96_syn_ms": _get(a96, "ref", "syn_ms"),
            "gm_fp32_ana_ms": _get(ag, "fp32", "ana_ms"),
            "gm_fp64_ana_ms": _get(ag, "fp64", "ana_ms"),
            "gm_fp32_syn_ms": _get(ag, "fp32", "syn_ms"),
            "gm_fp64_syn_ms": _get(ag, "fp64", "syn_ms"),
            "fp32_rel_dalm": _get(aa, "fp32", "rel_dalm"),
            "fp64_rel_dalm": _get(aa, "fp64", "rel_dalm"),
            "fp32_rms_dalm": _get(aa, "fp32", "rms_dalm"),
            "fp64_rms_dalm": _get(aa, "fp64", "rms_dalm"),
            "fp32_max_dalm": _get(aa, "fp32", "max_dalm"),
            "fp64_max_dalm": _get(aa, "fp64", "max_dalm"),
            "fp32_rel_dmap": _get(aa, "fp32", "rel_dmap"),
            "fp64_rel_dmap": _get(aa, "fp64", "rel_dmap"),
        }
        plot["sht"].append(row)
        md.append(
            f"| {ns} | {sp} | {fmt(row['ducc32_ana_ms'])} | {fmt(row['ducc96_ana_ms'])} | "
            f"{fmt(row['gm_fp32_ana_ms'])} | {fmt(row['gm_fp64_ana_ms'])} | "
            f"{sci(row['fp32_rel_dalm'])} | {sci(row['fp64_rel_dalm'])} | {sci(row['fp32_rms_dalm'])} |"
        )

    md += [
        "",
        "## 2. Full $C_\\ell$ pipeline (ms) and relative $\\Delta C_\\ell$",
        "",
        "| Nside | spin | NM 32c | NM 96c | GM fp32 | GM fp64 | fp32 rel dCl | fp64 rel dCl | fp32 rel coupled |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in pipe_keys:
        ns, sp = key
        a32, a96, ag, aa = p32.get(key, {}), p96.get(key, {}), pg.get(key, {}), pa.get(key, {})
        row = {
            "nside": ns,
            "spin": sp,
            "nm32_ms": _get(a32, "ref", "times", "total"),
            "nm96_ms": _get(a96, "ref", "times", "total"),
            "gm_fp32_ms": _get(ag, "fp32", "times", "total"),
            "gm_fp64_ms": _get(ag, "fp64", "times", "total"),
            "gm_fp32_stages_ms": _get(ag, "fp32", "times", default={}) if isinstance(_get(ag, "fp32", "times", default=None), dict) else {},
            "nm32_stages_ms": _get(a32, "ref", "times", default={}),
            "nm96_stages_ms": _get(a96, "ref", "times", default={}),
            "fp32_rel_dCl": _acc(ag, aa, "fp32", "rel_dCl"),
            "fp64_rel_dCl": _acc(ag, aa, "fp64", "rel_dCl"),
            "fp32_rms_rel_dCl": _acc(ag, aa, "fp32", "rms_rel_dCl"),
            "fp64_rms_rel_dCl": _acc(ag, aa, "fp64", "rms_rel_dCl"),
            "fp32_max_dCl": _acc(ag, aa, "fp32", "max_dCl"),
            "fp64_max_dCl": _acc(ag, aa, "fp64", "max_dCl"),
            "fp32_rel_coupled": _acc(ag, aa, "fp32", "rel_coupled_diff"),
            "fp64_rel_coupled": _acc(ag, aa, "fp64", "rel_coupled_diff"),
        }
        # stages as plain dicts of floats
        for name, src in (
            ("gm_fp32_stages_ms", _get(ag, "fp32", "times", default={})),
            ("gm_fp64_stages_ms", _get(ag, "fp64", "times", default={})),
            ("nm32_stages_ms", _get(a32, "ref", "times", default={})),
            ("nm96_stages_ms", _get(a96, "ref", "times", default={})),
        ):
            row[name] = src if isinstance(src, dict) else {}
        plot["pipeline"].append(row)
        md.append(
            f"| {ns} | {sp} | {fmt(row['nm32_ms'], 1)} | {fmt(row['nm96_ms'], 1)} | "
            f"{fmt(row['gm_fp32_ms'], 1)} | {fmt(row['gm_fp64_ms'], 1)} | "
            f"{sci(row['fp32_rel_dCl'])} | {sci(row['fp64_rel_dCl'])} | {sci(row['fp32_rel_coupled'])} |"
        )

    (root / "results_for_plots.json").write_text(json.dumps(plot, indent=2))
    (root / "results_cores_merged.json").write_text(json.dumps(plot, indent=2))
    (root / "tables_cores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nWrote", root / "results_for_plots.json")


if __name__ == "__main__":
    main()
