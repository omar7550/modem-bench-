"""Launch the dev campaign: both arms, sharded lanes per replicate, merged with arm-merge; restartable.

    ./.venv/bin/python scripts/launch_campaign.py --replicates 3 --shards 4 --lanes 4 --go
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
CAPTURES = REPO / "captures" / "dev-frozen-v1"
OUT_ROOT = REPO / "runs" / "campaign"

ARMS = ("best-of-n", "iterative")


def shard_command(arm: str, replicate: int, shard: int, shards: int) -> tuple[list[str], Path]:
    out = OUT_ROOT / f"replicate-{replicate}" / arm / f"shard-{shard}"
    command = [
        str(PYTHON), "-m", "modembench", "arm-run",
        "--arm", arm,
        "--split", "dev",
        "--captures-root", str(CAPTURES),
        "--transport", "subscription",
        "--shard", f"{shard}/{shards}",
        "--out", str(out),
    ]
    return command, out


def merged_path(arm: str, replicate: int) -> Path:
    return OUT_ROOT / f"replicate-{replicate}" / arm / "arm-run.json"


def run_replicate(replicate: int, *, shards: int, lanes: int, go: bool) -> None:
    for arm in ARMS:
        if merged_path(arm, replicate).is_file():
            print(f"replicate {replicate} {arm}: merged record exists, skipping")
            continue
        pending: list[tuple[list[str], Path]] = []
        for shard in range(shards):
            command, out = shard_command(arm, replicate, shard, shards)
            if (out / "arm-run.json").is_file():
                print(f"  {arm} r{replicate} shard {shard}: record exists, skipping")
                continue
            pending.append((command, out))
        if not go:
            for command, _ in pending:
                print("DRY:", " ".join(command))
            continue
        # Lanes run within one arm at a time to bound concurrency.
        running: list[tuple[subprocess.Popen, Path, list[str]]] = []
        while pending or running:
            while pending and len(running) < lanes:
                command, out = pending.pop(0)
                out.mkdir(parents=True, exist_ok=True)
                log = (out / "lane.log").open("ab")
                print("LAUNCH:", " ".join(command), flush=True)
                running.append(
                    (subprocess.Popen(command, stdout=log, stderr=log), out, command)
                )
            time.sleep(20)
            still = []
            for proc, out, command in running:
                code = proc.poll()
                if code is None:
                    still.append((proc, out, command))
                elif code != 0:
                    print(f"LANE FAILED ({code}): {' '.join(command)} — will not merge; "
                          "relaunch this script to retry the missing shard", flush=True)
                else:
                    print(f"LANE DONE: {out}", flush=True)
            running = still
        shard_records = [
            shard_command(arm, replicate, shard, shards)[1] / "arm-run.json"
            for shard in range(shards)
        ]
        if all(path.is_file() for path in shard_records):
            merge = [
                str(PYTHON), "-m", "modembench", "arm-merge",
                "--out", str(merged_path(arm, replicate)),
                *map(str, shard_records),
            ]
            print("MERGE:", " ".join(merge), flush=True)
            subprocess.run(merge, check=True)
        else:
            missing = [p for p in shard_records if not p.is_file()]
            print(f"replicate {replicate} {arm}: {len(missing)} shard(s) missing, merge "
                  "deferred — relaunch to retry", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()
    for replicate in range(args.replicates):
        run_replicate(replicate, shards=args.shards, lanes=args.lanes, go=args.go)
    done = {
        (arm, replicate): merged_path(arm, replicate).is_file()
        for arm in ARMS
        for replicate in range(args.replicates)
    }
    print(json.dumps({f"{arm}/r{rep}": ok for (arm, rep), ok in done.items()}, indent=1))
    if all(done.values()):
        print("\nCAMPAIGN COMPLETE. Next:")
        print("  modembench gate-analysis \\")
        print("    --one-shot", " ".join(str(merged_path("best-of-n", r)) for r in range(args.replicates)), "\\")
        print("    --iterative", " ".join(str(merged_path("iterative", r)) for r in range(args.replicates)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
