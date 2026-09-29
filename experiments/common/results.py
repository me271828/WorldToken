"""Compute paper tables from run logs and episode outcomes, including gzip files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from .configuration import EXPERIMENTS, configs, read_config

RMSE_KEY = "task_macro/action_rmse/stochastic/full10"
SCALES = (50, 100, 300, 1000, 2900)
PARAMETERS_M = {1: 44.3, 2: 85.3, 3: 218.8, 4: 648.9, 5: 1490.3}


def jsonl(path: Path):
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_table(output: Path, name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {name}")
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in fields) + " |")
    (output / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def episode_files(evaluation: Path) -> list[Path]:
    # A merged file takes precedence over its source shards.
    for base in (evaluation, evaluation / "full", evaluation / "tasks/blocks_ranking_try"):
        for suffix in (".jsonl", ".jsonl.gz"):
            path = base / ("episodes" + suffix)
            if path.is_file():
                return [path]
    paths = sorted((evaluation / "full/shards").glob("episodes_shard_*.jsonl*"))
    if not paths:
        raise FileNotFoundError(f"No episode results in {evaluation}")
    return paths


def success_rate(evaluation: Path, *, rmbench: bool = False) -> float:
    rows = [row for path in episode_files(evaluation) for row in jsonl(path)]
    expected = 100 if rmbench else 1150
    if len(rows) != expected:
        raise ValueError(f"{evaluation}: expected {expected} episodes, found {len(rows)}")
    ids = [r.get("candidate_seed") if rmbench else r.get("global_episode_id", (r.get("task"), r.get("episode_idx"))) for r in rows]
    if len(set(ids)) != expected:
        raise ValueError(f"Repeated episode identity in {evaluation}")
    if rmbench:
        return 100 * statistics.mean(bool(r["success"]) for r in rows)
    tasks = defaultdict(list)
    for row in rows:
        tasks[row.get("task", row.get("env_name"))].append(bool(row["success"]))
    if len(tasks) != 23 or any(len(v) != 50 for v in tasks.values()):
        raise ValueError(f"{evaluation}: expected 23 tasks × 50 episodes")
    return 100 * statistics.mean(statistics.mean(v) for v in tasks.values())


def final_rmse(run: Path, step: int) -> float:
    found = None
    for row in jsonl(run / "metrics.jsonl"):
        if row.get("event") == "holdout" and int(row.get("step", -1)) == step:
            value = row.get("metrics", {}).get(RMSE_KEY)
            if value is not None:
                found = float(value)
    if found is None:
        raise ValueError(f"No {RMSE_KEY} at final step {step} in {run}")
    return found


def recipes(section: str):
    for path in configs(section):
        cfg = read_config(path)
        yield cfg, cfg["paper"]


def run_result(root: Path, cfg: dict, paper: dict) -> dict:
    run = root / paper["section"] / paper["run"]
    context = int(paper.get("C_train", 10))
    rates = [success_rate(run / "rollouts" / f"C{context}_repeat{i:02d}") for i in (1, 2, 3)]
    return {"run": paper["run"], "D": paper.get("D", 300), "N": paper.get("N", "BC"),
            "seed": paper["seed"], "K": paper.get("K", "BC"), "C_train": context,
            "rmse": "" if paper["baseline"] else final_rmse(run, int(paper["target_step"])),
            "sr_percent": statistics.mean(rates),
            "sr_repeat1": rates[0], "sr_repeat2": rates[1], "sr_repeat3": rates[2]}


def scaling(root: Path, output: Path) -> None:
    rows = [run_result(root, cfg, p) for cfg, p in recipes("04")]
    grid = [r for r in rows if r["N"] != "BC"]
    write_table(output, "scaling_runs", grid)
    write_table(output, "bc_transformer", [r for r in rows if r["N"] == "BC"])
    means = []
    for n in range(1, 6):
        for d in SCALES:
            pair = [r for r in grid if r["N"] == n and r["D"] == d]
            if len(pair) != 2: raise ValueError(f"Expected two seeds at N{n}, D{d}")
            means.append(dict(N=n, D=d, parameters_m=PARAMETERS_M[n],
                              rmse=statistics.mean(r["rmse"] for r in pair),
                              sr_percent=statistics.mean(r["sr_percent"] for r in pair)))
    write_table(output, "scaling_seed_means", means)
    fits = []
    for n in range(1, 6):
        points = [r for r in means if r["N"] == n]
        x = [math.log(r["D"]) for r in points]; y = [math.log(r["rmse"]) for r in points]
        xm, ym = statistics.mean(x), statistics.mean(y)
        slope = sum((a-xm)*(b-ym) for a,b in zip(x,y)) / sum((a-xm)**2 for a in x)
        intercept = ym - slope*xm
        residual = sum((b-(intercept+slope*a))**2 for a,b in zip(x,y))
        fits.append(dict(N=n, a=math.exp(intercept), alpha=-slope,
                         r2_log_space=1-residual/sum((b-ym)**2 for b in y)))
    write_table(output, "scaling_power_laws", fits)


def temporal_flops(context: int, tokens: int) -> int:
    length = context * tokens
    # Four Qwen2 layers, d=768, FFN=2048; one MAC counts as two FLOPs.
    return 56_623_104 * length + 12_288 * length * length


def token(root: Path, output: Path) -> None:
    selected = [(cfg,p) for cfg,p in recipes("04") if not p["baseline"] and p["N"] == 2]
    selected += list(recipes("05"))
    rows = []
    for cfg, paper in selected:
        row = run_result(root, cfg, paper); k = row["K"]
        row.update(parameters_m={1:85.3,4:92.4,50:83.0}[k], temporal_flops=temporal_flops(10,k),
                   relative_temporal_flops=temporal_flops(10,k)/temporal_flops(10,1))
        rows.append(row)
    write_table(output, "token_interface", rows)
    reference = {r["D"]:r for r in rows if r["K"]==1 and r["seed"]==0}
    write_table(output, "token_changes_seed0", [dict(D=r["D"],K=r["K"],
        sr_change_pp=r["sr_percent"]-reference[r["D"]]["sr_percent"],
        rmse_change_percent=100*(r["rmse"]/reference[r["D"]]["rmse"]-1)) for r in rows if r["K"]!=1])


def history(root: Path, output: Path) -> None:
    truncation = []; matched = []
    for cfg, paper in recipes("04"):
        if paper["baseline"]: continue
        ref = run_result(root,cfg,paper)
        run = root/paper["section"]/paper["run"]
        for c in (1,2,5):
            sr = success_rate(run/"rollouts"/f"C{c}_repeat01")
            truncation.append(dict(run=paper["run"],D=paper["D"],N=paper["N"],seed=paper["seed"],
                                   C_test=c,sr_percent=sr,reference_c10_sr=ref["sr_percent"],
                                   delta_sr_pp=sr-ref["sr_percent"]))
        if paper["D"] == 300 and paper["N"] == 3: matched.append(ref)
    matched += [run_result(root,cfg,p) for cfg,p in recipes("06")]
    write_table(output,"history_truncation",truncation)
    write_table(output,"matched_training_history",matched)


def ranking(root: Path, output: Path, behavior: bool) -> None:
    paper = next(p for _,p in recipes("07") if p["target_step"]==5500)
    run = root/paper["section"]/paper["run"]
    rows = [dict(C_test=c,history_seconds=c*.24,sr_percent=success_rate(run/"rollouts"/f"C{c}_100eps",rmbench=True))
            for c in (608,288,128,64,32)]
    write_table(output,"ranking_success",rows)
    stress = []
    for path in sorted((run/"rollouts").glob("stress_seed*")):
        # Correct swaps are the confirmed sequence events, not environment successes.
        recorded_events = list(jsonl(path/"sequence_events.jsonl"))
        initial = next(e["order"] for e in recorded_events if e.get("event")=="initial_stable_order")
        order = list(initial); required = 0
        for a,b in ((1,2),(0,2),(0,1),(1,2),(0,2)):
            order[a],order[b] = order[b],order[a]; required += 1
            if order == [1,2,3]: break
        if order != [1,2,3]: raise ValueError(f"Invalid initial order in {path}")
        events = [event for event in recorded_events if event.get("event")=="confirmed_expert_swap"]
        summary = read_config(path/"summary.json")
        last = max((float(e["simulated_seconds"]) for e in events),default=0.)
        stress.append(dict(seed=int(path.name.removeprefix("stress_seed")),
                           swaps_to_first_target=required,
                           correct_swaps=max((int(e["completed_swaps"]) for e in events),default=0),
                           last_correct_swap_seconds=last,history_window_seconds=608*.24,
                           correct_swap_after_window_slides=last>608*.24,
                           trajectory_seconds=float(summary["simulated_seconds"])))
    if len(stress)!=9: raise ValueError("Expected the nine paper continuation trajectories")
    write_table(output,"ranking_stress",stress)
    if behavior:
        scripts = EXPERIMENTS/paper["section"]/'analysis'
        for name in ("audit_ranking_success_behavior", "audit_phase_confusion_signature",
                     "audit_failure_execution_factors", "audit_swap_cycle_duration"):
            subprocess.run([sys.executable,str(scripts/f"{name}.py"),"--rollouts-root",str(run/"rollouts"),
                            "--output-dir",str(output)],check=True)


def main(section: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if section is None: parser.add_argument("--section",choices=("04","05","06","07"),required=True)
    parser.add_argument("--records-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path("results"))
    parser.add_argument("--behavior",action="store_true",help="Also run the four RMBench behavior analyses")
    args = parser.parse_args()
    selected = section or args.section
    output = args.output_dir.resolve()
    if selected == "07": ranking(args.records_root.resolve(),output,args.behavior)
    else: {"04":scaling,"05":token,"06":history}[selected](args.records_root.resolve(),output)
    print(f"Wrote Section {selected} tables to {output}")


if __name__ == "__main__":
    main()
