#!/usr/bin/env python3
"""Checkpointable model server used only by E4 long-horizon rollouts."""

from __future__ import annotations

import hashlib
import os
import signal
from collections import deque
from pathlib import Path
from typing import Any

import torch

from experiments.rmbench_tools.rmbench_model_server import RMBenchModelServer, parse_args


POLICY_STATE_SCHEMA = "rmbench_long_horizon_policy_state_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_policy_state(policy: Any, *, checkpoint_global_step: int) -> dict[str, Any]:
    if not bool(policy.cache_encoded_history):
        raise ValueError("long-horizon checkpoints require cache_encoded_history=True")
    encoded = [tensor.detach().to(device="cpu") for tensor in policy._encoded_history]
    if len(encoded) != policy.history_length:
        raise RuntimeError("policy raw and encoded history lengths differ")
    return {
        "schema": POLICY_STATE_SCHEMA,
        "checkpoint_global_step": int(checkpoint_global_step),
        "task_name": str(policy.task_name),
        "seed": int(policy.seed),
        "precision": str(policy.precision),
        "max_history": int(policy.max_history),
        "execute_steps": int(policy.execute_steps),
        "deterministic": bool(policy.deterministic),
        "cache_encoded_history": True,
        "history_length": int(policy.history_length),
        "encoded_history": encoded,
        "generator_state": policy.generator.get_state().to(device="cpu"),
        "intermediate_update_count": int(policy.intermediate_update_count),
    }


def restore_policy_state(policy: Any, state: dict[str, Any], *, expected_step: int) -> None:
    if state.get("schema") != POLICY_STATE_SCHEMA:
        raise ValueError("unexpected long-horizon policy-state schema")
    expected = {
        "checkpoint_global_step": int(expected_step),
        "task_name": str(policy.task_name),
        "seed": int(policy.seed),
        "precision": str(policy.precision),
        "max_history": int(policy.max_history),
        "execute_steps": int(policy.execute_steps),
        "deterministic": bool(policy.deterministic),
        "cache_encoded_history": True,
    }
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"policy checkpoint/config mismatch: {mismatches}")
    encoded = list(state["encoded_history"])
    history_length = int(state["history_length"])
    if len(encoded) != history_length or history_length > int(policy.max_history):
        raise ValueError("invalid encoded history length in policy checkpoint")
    policy._history = deque(({} for _ in range(history_length)), maxlen=policy.max_history)
    policy._encoded_history = deque(
        (tensor.to(device=policy.device) for tensor in encoded),
        maxlen=policy.max_history,
    )
    policy.generator.set_state(state["generator_state"])
    policy.intermediate_update_count = int(state["intermediate_update_count"])


class LongHorizonModelServer(RMBenchModelServer):
    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("cmd", ""))
        if command == "save_session":
            session_id = str(request["session_id"])
            path = Path(request["path"]).resolve()
            with self.sessions_lock:
                policy = self.sessions.get(session_id)
            if policy is None:
                raise KeyError(f"unknown session {session_id!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with self.model_lock:
                payload = export_policy_state(
                    policy,
                    checkpoint_global_step=self.checkpoint_global_step,
                )
                torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
            return {
                "ok": True,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "history_length": policy.history_length,
            }
        if command == "restore_session":
            session_id = str(request["session_id"])
            path = Path(request["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            with self.model_lock:
                policy = self._new_policy(
                    task_name=str(state["task_name"]),
                    seed=int(state["seed"]),
                )
                restore_policy_state(
                    policy,
                    state,
                    expected_step=self.checkpoint_global_step,
                )
            with self.sessions_lock:
                self.sessions[session_id] = policy
            self.max_history_seen = max(self.max_history_seen, policy.history_length)
            self._write_status("ready")
            return {
                "ok": True,
                "session_id": session_id,
                "history_length": policy.history_length,
                "sha256": _sha256(path),
            }
        return super().dispatch(request)


def main() -> int:
    args = parse_args()
    if args.max_cached_memory_mib < 0:
        raise ValueError("--max-cached-memory-mib must be >= 0")
    if args.min_free_memory_mib < 0:
        raise ValueError("--min-free-memory-mib must be >= 0")
    if not args.cache_encoded_history:
        raise ValueError("long-horizon server requires --cache-encoded-history")
    server = LongHorizonModelServer(args)

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
