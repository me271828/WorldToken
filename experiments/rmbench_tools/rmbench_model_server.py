#!/usr/bin/env python3
"""Local, session-aware RMBench model server.

The server owns one WorldToken model on one GPU. Each connected simulator
uses an independent RMBenchRolloutPolicy history and diffusion RNG while
sharing the frozen model weights. Requests are serialized around model
inference so stateful histories cannot cross sessions and concurrent CUDA calls
cannot inflate memory unpredictably.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import torch

from worldtoken.rmbench_policy import (
    RMBenchRolloutPolicy,
    load_rmbench_rollout_policy,
)
from worldtoken.train_utils import load_checkpoint


HEADER = struct.Struct("!Q")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("socket closed before the framed message completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    size = HEADER.unpack(recv_exact(sock, HEADER.size))[0]
    if size > 64 * 1024 * 1024:
        raise ValueError(f"refusing oversized RPC payload: {size} bytes")
    return pickle.loads(recv_exact(sock, size))


def send_message(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=5)
    sock.sendall(HEADER.pack(len(data)))
    sock.sendall(data)


class RMBenchModelServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = load_checkpoint(Path(args.checkpoint), map_location="cpu")
        self.checkpoint_global_step = int(checkpoint.get("global_step", -1))
        if self.checkpoint_global_step != int(args.expected_step):
            raise ValueError(f"checkpoint global_step={self.checkpoint_global_step}, expected {args.expected_step}")
        del checkpoint

        started = time.perf_counter()
        bootstrap = load_rmbench_rollout_policy(
            args.checkpoint,
            task_name="battery_try",
            device=args.device,
            precision=args.precision,
            max_history=args.max_history,
            execute_steps=args.execute_steps,
            deterministic=args.deterministic,
            seed=0,
            cache_encoded_history=args.cache_encoded_history,
        )
        self.model = bootstrap.model
        self.device = bootstrap.device
        del bootstrap
        self.load_seconds = time.perf_counter() - started

        self.sessions: dict[str, RMBenchRolloutPolicy] = {}
        self.sessions_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.server_socket: socket.socket | None = None
        self.started_unix = time.time()
        self.action_requests = 0
        self.update_requests = 0
        self.max_history_seen = 0
        self.total_inference_seconds = 0.0
        self.max_inference_seconds = 0.0
        self.total_queue_seconds = 0.0
        self.max_queue_seconds = 0.0
        self.total_compute_seconds = 0.0
        self.max_compute_seconds = 0.0
        self.cache_purges = 0
        self.total_cache_purge_seconds = 0.0
        self.last_cache_purge_unix: float | None = None
        self.last_cache_purge_reason: str | None = None
        self._write_status("ready")

    def _status_payload(self, status: str) -> dict[str, Any]:
        cuda = self.device.type == "cuda"
        allocated_mib = torch.cuda.memory_allocated(self.device) / 2**20 if cuda else None
        reserved_mib = torch.cuda.memory_reserved(self.device) / 2**20 if cuda else None
        free_mib = torch.cuda.mem_get_info(self.device)[0] / 2**20 if cuda else None
        return {
            "schema": "rmbench_model_server_v1",
            "status": status,
            "host": self.args.host,
            "port": self.args.port,
            "checkpoint": str(Path(self.args.checkpoint).resolve()),
            "checkpoint_global_step": self.checkpoint_global_step,
            "device": str(self.device),
            "precision": self.args.precision,
            "deterministic": bool(self.args.deterministic),
            "max_history": int(self.args.max_history),
            "execute_steps": int(self.args.execute_steps),
            "cache_encoded_history": bool(self.args.cache_encoded_history),
            "max_cached_memory_mib": float(self.args.max_cached_memory_mib),
            "min_free_memory_mib": float(self.args.min_free_memory_mib),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "load_seconds": self.load_seconds,
            "uptime_seconds": time.time() - self.started_unix,
            "session_count": len(self.sessions),
            "action_requests": self.action_requests,
            "update_requests": self.update_requests,
            "max_history_seen": self.max_history_seen,
            "mean_inference_seconds": (
                self.total_inference_seconds / self.action_requests if self.action_requests else None
            ),
            "max_inference_seconds": (self.max_inference_seconds if self.action_requests else None),
            "mean_queue_seconds": (self.total_queue_seconds / self.action_requests if self.action_requests else None),
            "max_queue_seconds": (self.max_queue_seconds if self.action_requests else None),
            "mean_compute_seconds": (
                self.total_compute_seconds / self.action_requests if self.action_requests else None
            ),
            "max_compute_seconds": (self.max_compute_seconds if self.action_requests else None),
            "cache_purges": self.cache_purges,
            "total_cache_purge_seconds": self.total_cache_purge_seconds,
            "last_cache_purge_unix": self.last_cache_purge_unix,
            "last_cache_purge_reason": self.last_cache_purge_reason,
            "cuda_memory_allocated_mib": (round(allocated_mib, 3) if allocated_mib is not None else None),
            "cuda_memory_reserved_mib": (round(reserved_mib, 3) if reserved_mib is not None else None),
            "cuda_memory_cached_mib": (
                round(max(reserved_mib - allocated_mib, 0.0), 3)
                if reserved_mib is not None and allocated_mib is not None
                else None
            ),
            "cuda_memory_free_mib": (round(free_mib, 3) if free_mib is not None else None),
            "cuda_peak_memory_allocated_mib": (
                round(torch.cuda.max_memory_allocated(self.device) / 2**20, 3) if cuda else None
            ),
            "updated_unix": time.time(),
        }

    def _maybe_release_cuda_cache(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {"purged": False, "reason": None, "seconds": 0.0}

        allocated_mib = torch.cuda.memory_allocated(self.device) / 2**20
        reserved_mib = torch.cuda.memory_reserved(self.device) / 2**20
        cached_mib = max(reserved_mib - allocated_mib, 0.0)
        free_mib = torch.cuda.mem_get_info(self.device)[0] / 2**20
        reasons: list[str] = []
        if self.args.max_cached_memory_mib > 0 and cached_mib >= self.args.max_cached_memory_mib:
            reasons.append(f"cached_mib={cached_mib:.1f}>={self.args.max_cached_memory_mib:.1f}")
        if self.args.min_free_memory_mib > 0 and free_mib <= self.args.min_free_memory_mib:
            reasons.append(f"free_mib={free_mib:.1f}<={self.args.min_free_memory_mib:.1f}")
        if not reasons:
            return {"purged": False, "reason": None, "seconds": 0.0}

        started = time.perf_counter()
        torch.cuda.empty_cache()
        purge_seconds = time.perf_counter() - started
        reason = ";".join(reasons)
        self.cache_purges += 1
        self.total_cache_purge_seconds += purge_seconds
        self.last_cache_purge_unix = time.time()
        self.last_cache_purge_reason = reason
        return {
            "purged": True,
            "reason": reason,
            "seconds": purge_seconds,
        }

    def _write_status(self, status: str) -> None:
        with self.stats_lock:
            payload = self._status_payload(status)
            path = self.output_dir / "model_server_status.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

    def _new_policy(
        self,
        *,
        task_name: str,
        seed: int,
    ) -> RMBenchRolloutPolicy:
        return RMBenchRolloutPolicy(
            self.model,
            task_name=task_name,
            device=self.device,
            precision=self.args.precision,
            max_history=self.args.max_history,
            execute_steps=self.args.execute_steps,
            deterministic=self.args.deterministic,
            seed=int(seed),
            cache_encoded_history=self.args.cache_encoded_history,
        )

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("cmd", ""))
        if command == "ping":
            return {
                "ok": True,
                "status": "ready",
                "checkpoint_global_step": self.checkpoint_global_step,
            }
        if command == "reset":
            session_id = str(request["session_id"])
            with self.model_lock:
                policy = self._new_policy(
                    task_name=str(request["task_name"]),
                    seed=int(request["seed"]),
                )
            with self.sessions_lock:
                self.sessions[session_id] = policy
            self._write_status("ready")
            return {"ok": True, "session_id": session_id, "history_length": 0}
        if command == "close_session":
            session_id = str(request["session_id"])
            with self.sessions_lock:
                self.sessions.pop(session_id, None)
            self._write_status("ready")
            return {"ok": True, "session_id": session_id}
        if command == "update_obs":
            session_id = str(request["session_id"])
            with self.sessions_lock:
                policy = self.sessions.get(session_id)
            if policy is None:
                raise KeyError(f"unknown session {session_id!r}")
            policy.update_obs()
            self.update_requests += 1
            return {
                "ok": True,
                "intermediate_update_count": policy.intermediate_update_count,
            }
        if command == "get_action":
            session_id = str(request["session_id"])
            with self.sessions_lock:
                policy = self.sessions.get(session_id)
            if policy is None:
                raise KeyError(f"unknown session {session_id!r}")
            started = time.perf_counter()
            with self.model_lock:
                queue_seconds = time.perf_counter() - started
                compute_started = time.perf_counter()
                action = policy.get_action(request["observation"])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                compute_seconds = time.perf_counter() - compute_started
                cache_purge = self._maybe_release_cuda_cache()
            elapsed = time.perf_counter() - started
            self.action_requests += 1
            self.total_inference_seconds += elapsed
            self.max_inference_seconds = max(self.max_inference_seconds, elapsed)
            self.total_queue_seconds += queue_seconds
            self.max_queue_seconds = max(self.max_queue_seconds, queue_seconds)
            self.total_compute_seconds += compute_seconds
            self.max_compute_seconds = max(self.max_compute_seconds, compute_seconds)
            self.max_history_seen = max(self.max_history_seen, policy.history_length)
            self._write_status("ready")
            return {
                "ok": True,
                "actions": action,
                "history_length": policy.history_length,
                "inference_seconds": elapsed,
                "queue_seconds": queue_seconds,
                "compute_seconds": compute_seconds,
                "cache_purge": cache_purge,
            }
        if command == "shutdown":
            self.shutdown_event.set()
            if self.server_socket is not None:
                self.server_socket.close()
            return {"ok": True}
        raise ValueError(f"unknown command {command!r}")

    def handle_client(self, client: socket.socket, address: Any) -> None:
        del address
        with client:
            while not self.shutdown_event.is_set():
                try:
                    request = recv_message(client)
                except ConnectionError:
                    return
                try:
                    response = self.dispatch(request)
                except Exception as exc:
                    response = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                try:
                    send_message(client, response)
                except (BrokenPipeError, ConnectionResetError):
                    return

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self.server_socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.args.host, self.args.port))
            server.listen(64)
            server.settimeout(1.0)
            print(
                json.dumps(
                    {
                        "event": "rmbench_model_server_ready",
                        "host": self.args.host,
                        "port": self.args.port,
                        "checkpoint_global_step": self.checkpoint_global_step,
                        "load_seconds": self.load_seconds,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            while not self.shutdown_event.is_set():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.shutdown_event.is_set():
                        break
                    raise
                threading.Thread(
                    target=self.handle_client,
                    args=(client, address),
                    daemon=True,
                ).start()
        self._write_status("stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max-history", type=int, default=288)
    parser.add_argument("--execute-steps", type=int, default=4)
    parser.add_argument(
        "--cache-encoded-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache per-observation bottleneck z values instead of re-encoding history.",
    )
    parser.add_argument(
        "--max-cached-memory-mib",
        type=float,
        default=8192.0,
        help="Release the CUDA allocator cache after a request above this size; 0 disables.",
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=float,
        default=16384.0,
        help="Release the CUDA allocator cache below this device-free threshold; 0 disables.",
    )
    sampler = parser.add_mutually_exclusive_group()
    sampler.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
    )
    sampler.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
    )
    parser.set_defaults(deterministic=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cached_memory_mib < 0:
        raise ValueError("--max-cached-memory-mib must be >= 0")
    if args.min_free_memory_mib < 0:
        raise ValueError("--min-free-memory-mib must be >= 0")
    server = RMBenchModelServer(args)

    def stop(_signum: int, _frame: Any) -> None:
        server.shutdown_event.set()
        if server.server_socket is not None:
            server.server_socket.close()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
