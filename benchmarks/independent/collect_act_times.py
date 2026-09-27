"""Fold ACT warm RESULT lines into act_warm_times.json."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "act_warm_times.json"
LOGS = Path("/home/lxu/.cursor/projects/home-lxu-scratch-agent-dev-auto-research-agent-GMaster/terminals")


def rows_from(text):
    found = []
    for line in text.splitlines():
        i = line.find("RESULT {")
        if i < 0:
            continue
        try:
            row = json.loads(line[i + 7 :])
        except json.JSONDecodeError:
            continue
        if "act_cl_" in row.get("cl_path", ""):
            found.append(row)
    return found


def main():
    rows = []
    if LOGS.is_dir():
        for path in LOGS.glob("*.txt"):
            rows.extend(rows_from(path.read_text(errors="replace")))
    proc = subprocess.run(
        ["sg", "docker", "-c", "docker ps -aq --filter ancestor=denario-sandbox:cpu"],
        capture_output=True, text=True,
    )
    for cid in proc.stdout.split():
        log = subprocess.run(
            ["sg", "docker", "-c", f"docker logs {cid}"],
            capture_output=True, text=True,
        )
        rows.extend(rows_from(log.stdout + log.stderr))
    keep = {}
    for row in rows:
        keep[(row["engine"], row["cores"], row["nside"], row["spin"])] = row
    runs = sorted(keep.values(), key=lambda r: (r["engine"], r["cores"], r["spin"], r["nside"]))
    OUT.write_text(json.dumps({
        "note": "Five warm wall times per cell, seconds. This bench did not record memory.",
        "runs": runs,
    }, indent=2) + "\n")
    print(f"wrote {len(runs)} {OUT}")


if __name__ == "__main__":
    main()
