"""Launch the paper's formal history sweep or a continuation trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from experiments.common.configuration import CODE_ROOT, child_environment

TOOLS = "experiments.rmbench_tools."
TASK = "blocks_ranking_try"
STRESS_SEEDS = (100000,100001,100002,100003,100004,100007,100008,100015,100016)


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()


def run_evaluation(args, history: int, stress_seed: int | None) -> None:
    output = args.output_dir.resolve() / (f"stress_seed{stress_seed}" if stress_seed is not None else f"C{history}_100eps")
    selection = Path(__file__).with_name("seed_selection.json").resolve()
    model_name = "rmbench_long_horizon_model_server" if stress_seed is not None else "rmbench_model_server"
    server = [sys.executable,"-m",TOOLS+model_name,"--checkpoint",str(args.checkpoint.resolve()),
              "--expected-step","5500","--output-dir",str(output/"model_server"),"--port",str(args.port),
              "--device",args.device,"--precision","bf16","--max-history",str(history),
              "--execute-steps","4","--cache-encoded-history","--stochastic"]
    workers = []
    if stress_seed is not None:
        workers.append([args.env_python,"-m",TOOLS+"rmbench_long_horizon_worker","--seed-selection",str(selection),
                        "--eligible-index",str(stress_seed-100000),"--output-dir",str(output),"--server-port",str(args.port),
                        "--stable-confirmations","8","--stall-action-steps","1500",
                        "--segment-action-steps","1000","--checkpoint-action-steps","1000",
                        "--min-free-disk-gib",str(args.min_free_disk_gib),"--video-fps","50/3"])
        if args.resume: workers[0].append("--resume")
    else:
        for index in range(args.workers):
            start, end = index*100//args.workers, (index+1)*100//args.workers
            worker_dir = output/"tasks"/TASK/"shards"/f"shard_{index:03d}"
            cmd = [args.env_python,"-m",TOOLS+"rmbench_seeded_rollout_worker","--task",TASK,
                   "--seed-selection",str(selection),"--output-dir",str(worker_dir),"--server-port",str(args.port),
                   "--selection-start",str(start),"--selection-stop",str(end),"--instruction-type","unseen",
                   "--save-videos","100","--video-fps","50/3","--blocks-failure-mode","off",
                   "--blocks-success-diagnostics"]
            if args.resume: cmd.append("--resume-existing")
            workers.append(cmd)
    for cmd in [server,*workers]: print(shlex.join(cmd),flush=True)
    if args.print_command: return
    output.mkdir(parents=True,exist_ok=True)
    env = child_environment({"RMBENCH_ROOT":str(args.rmbench_root.resolve())})
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    status_file = output/"model_server/model_server_status.json"
    if status_file.exists(): status_file.unlink()
    processes = []; logs = []
    with (output/"model_server.log").open("w",encoding="utf-8") as server_log:
        model = subprocess.Popen(server,cwd=CODE_ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+600
            while True:
                if model.poll() is not None: raise RuntimeError(f"Model server failed; see {output/'model_server.log'}")
                if status_file.exists():
                    state = json.loads(status_file.read_text(encoding="utf-8"))
                    if state.get("status")=="ready":
                        try:
                            with socket.create_connection(("127.0.0.1",args.port),timeout=1):
                                break
                        except OSError:
                            pass
                if time.monotonic()>deadline: raise TimeoutError("Model server did not become ready")
                time.sleep(1)
            for index, cmd in enumerate(workers):
                log=(output/f"worker_{index:03d}.log").open("w",encoding="utf-8");logs.append(log)
                processes.append(subprocess.Popen(cmd,cwd=CODE_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                if model.poll() is not None: raise RuntimeError("Model server exited during rollout")
                if any(p.poll() not in (None,0) for p in processes): raise RuntimeError(f"Rollout worker failed; see {output}")
                time.sleep(1)
            if any(p.returncode for p in processes): raise RuntimeError(f"Rollout worker failed; see {output}")
        finally:
            for p in processes: stop(p)
            stop(model)
            for log in logs: log.close()
    if stress_seed is None:
        from experiments.rmbench_tools.aggregate_rmbench_sharded_rollout import aggregate_task
        aggregate_task(output_dir=output,selection=json.loads(selection.read_text(encoding="utf-8")),
                       task=TASK,expected_shards=args.workers)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--rmbench-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True,help="The final training run's rollouts directory")
    parser.add_argument("--env-python",default=sys.executable,help="Python in the RMBench simulator environment")
    parser.add_argument("--history",type=int,choices=(608,288,128,64,32))
    parser.add_argument("--stress-seed",type=int,choices=STRESS_SEEDS)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--port",type=int,default=45210)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--min-free-disk-gib",type=float,default=200.)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--print-command",action="store_true")
    args=parser.parse_args()
    if not 1<=args.workers<=100: parser.error("--workers must be between 1 and 100")
    if args.stress_seed is not None:
        if args.history not in (None,608): parser.error("Continuation experiments use C=608")
        run_evaluation(args,608,args.stress_seed)
    else:
        for c in ([args.history] if args.history else [608,288,128,64,32]):
            run_evaluation(args,c,None)


if __name__ == "__main__":
    main()
