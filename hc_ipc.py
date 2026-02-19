#!/usr/bin/env python3
"""UDP IPC helper for HyperCube audio updates."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class HyperCubeUpdate:
    base_ops: List[float]
    levels: List[float]
    mod_matrix: Optional[List[List[float]]] = None
    input_matrix: Optional[List[List[float]]] = None
    mix_mode: str = "concurrent"


class HcIpcClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 50007) -> None:
        self.host = host
        self.port = port

    def _send(self, payload: dict) -> None:
        message = json.dumps(payload).encode("utf8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message, (self.host, self.port))
        sock.close()

    def send_update(self, update: HyperCubeUpdate) -> None:
        payload = {
            "cmd": "update",
            "base_ops": list(update.base_ops),
            "levels": list(update.levels),
            "mix_mode": update.mix_mode,
        }
        if update.mod_matrix is not None:
            payload["mod_matrix"] = update.mod_matrix
        if update.input_matrix is not None:
            payload["input_matrix"] = update.input_matrix
        self._send(payload)

    def send_state(
        self,
        base_ops: Sequence[float],
        levels: Sequence[float],
        mod_matrix: Optional[Sequence[Sequence[float]]] = None,
        input_matrix: Optional[Sequence[Sequence[float]]] = None,
        mix_mode: str = "concurrent",
    ) -> None:
        update = HyperCubeUpdate(
            base_ops=list(base_ops),
            levels=list(levels),
            mod_matrix=[list(row) for row in mod_matrix] if mod_matrix is not None else None,
            input_matrix=[list(row) for row in input_matrix] if input_matrix is not None else None,
            mix_mode=mix_mode,
        )
        self.send_update(update)

    def start(self) -> None:
        self._send({"cmd": "start"})

    def stop(self) -> None:
        self._send({"cmd": "stop"})

    def quit(self) -> None:
        self._send({"cmd": "quit"})
