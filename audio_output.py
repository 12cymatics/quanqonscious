#!/usr/bin/env python3
"""
Standalone audio synth for QuanQonscious implementing an FM-style HyperCube engine.

Features
- Operators with ratio/detune/level parameters.
- Modulation matrix and input matrix for frequency control.
- Three mix modes: concurrent, parallel, serial.
- UDP control via JSON messages on localhost:50007.

Message format examples (JSON):
  {"cmd":"update","base_ops":[110,220],"levels":[0.4,0.3],
   "mod_matrix":[[0,0.1],[0,0]],"input_matrix":[[0.0],[0.02]],
   "mix_mode":"concurrent"}
  {"cmd":"start"}, {"cmd":"stop"}, {"cmd":"quit"}
"""

from __future__ import annotations

import importlib.util
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

SAMPLE_RATE = 48000
BLOCKSIZE = 256
UDP_PORT = 50007
UDP_ADDR = ("127.0.0.1", UDP_PORT)


def _optional_import(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


if _optional_import("sounddevice"):
    import sounddevice as sd
    AUDIO_BACKEND = "sounddevice"
else:
    sd = None
    AUDIO_BACKEND = "simpleaudio" if _optional_import("simpleaudio") else "none"

if _optional_import("simpleaudio"):
    import simpleaudio as sa
else:
    sa = None


@dataclass
class Operator:
    base_freq: float
    level: float
    ratio: float = 1.0
    detune: float = 0.0
    phase: float = 0.0


class HyperCubeAudio:
    def __init__(self, num_ops: int = 12) -> None:
        self.num_ops = max(1, int(num_ops))
        self.ops: List[Operator] = [
            Operator(base_freq=110.0 + 55.0 * i, level=0.15) for i in range(self.num_ops)
        ]
        self.mod_matrix = np.zeros((self.num_ops, self.num_ops), dtype=np.float32)
        self.input_matrix = np.zeros((self.num_ops, 1), dtype=np.float32)
        self.mix_mode = "concurrent"
        self.running = False
        self.lock = threading.Lock()
        self.master_gain = 0.25

    def set_base_ops(self, bases: Sequence[float]) -> None:
        with self.lock:
            for i, freq in enumerate(bases[: self.num_ops]):
                self.ops[i].base_freq = float(freq)

    def set_levels(self, levels: Sequence[float]) -> None:
        with self.lock:
            for i, level in enumerate(levels[: self.num_ops]):
                self.ops[i].level = float(max(0.0, min(1.0, level)))

    def set_mod_matrix(self, mat: Sequence[Sequence[float]]) -> None:
        with self.lock:
            if len(mat) != self.num_ops:
                return
            if not all(len(row) == self.num_ops for row in mat):
                return
            self.mod_matrix = np.array(mat, dtype=np.float32)

    def set_input_matrix(self, mat: Sequence[Sequence[float]]) -> None:
        with self.lock:
            if len(mat) == 0:
                return
            rows = []
            for row in mat:
                if isinstance(row, (list, tuple, np.ndarray)):
                    rows.append([float(v) for v in row])
                else:
                    rows.append([float(row)])
            self.input_matrix = np.array(rows, dtype=np.float32)

    def set_mix_mode(self, mode: str) -> None:
        if mode in ("concurrent", "parallel", "serial"):
            with self.lock:
                self.mix_mode = mode

    def _base_frequencies(self, ops: List[Operator]) -> np.ndarray:
        freqs = np.zeros(self.num_ops, dtype=np.float32)
        for i, op in enumerate(ops):
            base = op.base_freq * op.ratio + op.detune
            if i < self.input_matrix.shape[0]:
                base += float(np.mean(self.input_matrix[i]))
            freqs[i] = max(20.0, min(18000.0, base))
        return freqs

    def _render_operator(self, op: Operator, freq_mod: np.ndarray) -> np.ndarray:
        phases = np.zeros_like(freq_mod, dtype=np.float32)
        phase = op.phase
        for i, f_mod in enumerate(freq_mod):
            freq = max(20.0, min(18000.0, op.base_freq * op.ratio + op.detune + f_mod))
            phase += 2.0 * math.pi * freq / SAMPLE_RATE
            phases[i] = phase
        op.phase = phase % (2.0 * math.pi)
        return np.sin(phases).astype(np.float32) * op.level

    def compute_frame(self, n_frames: int) -> np.ndarray:
        out = np.zeros(n_frames, dtype=np.float32)
        with self.lock:
            ops_copy = [Operator(**op.__dict__) for op in self.ops]
            mod = self.mod_matrix.copy()
            mix_mode = self.mix_mode
            input_matrix = self.input_matrix.copy()

        base_freqs = self._base_frequencies(ops_copy)

        if mix_mode == "concurrent":
            base_outs = []
            for i, op in enumerate(ops_copy):
                op.base_freq = base_freqs[i]
                base_outs.append(self._render_operator(op, np.zeros(n_frames, dtype=np.float32)))

            for i, op in enumerate(ops_copy):
                freq_mod = np.zeros(n_frames, dtype=np.float32)
                for j in range(self.num_ops):
                    if mod[i, j] != 0.0:
                        freq_mod += mod[i, j] * base_outs[j]
                if i < input_matrix.shape[0]:
                    freq_mod += float(np.mean(input_matrix[i]))
                op.base_freq = base_freqs[i]
                out += self._render_operator(op, freq_mod)

        elif mix_mode == "parallel":
            for i, op in enumerate(ops_copy):
                freq_mod = np.zeros(n_frames, dtype=np.float32)
                for j in range(i + 1):
                    freq_mod += mod[i, j] * out
                if i < input_matrix.shape[0]:
                    freq_mod += float(np.mean(input_matrix[i]))
                op.base_freq = base_freqs[i]
                out += self._render_operator(op, freq_mod)

        else:
            prev = np.zeros(n_frames, dtype=np.float32)
            for i, op in enumerate(ops_copy):
                freq_mod = mod[i, i - 1] * prev if i > 0 else np.zeros(n_frames, dtype=np.float32)
                if i < input_matrix.shape[0]:
                    freq_mod += float(np.mean(input_matrix[i]))
                op.base_freq = base_freqs[i]
                current = self._render_operator(op, freq_mod)
                out += current
                prev = current

        out = np.tanh(out * self.master_gain)
        return out

    def udp_listener(self, host: str = "127.0.0.1", port: int = UDP_PORT) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(0.5)
        print(f"[audio_output] Listening UDP {host}:{port}")
        while True:
            try:
                data, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            msg = json.loads(data.decode("utf8"))
            cmd = msg.get("cmd")
            if cmd == "quit":
                print("[audio_output] quit received")
                self.running = False
                break
            if cmd == "start":
                self.running = True
            if cmd == "stop":
                self.running = False
            if cmd == "update":
                if "base_ops" in msg:
                    self.set_base_ops(msg["base_ops"])
                if "levels" in msg:
                    self.set_levels(msg["levels"])
                if "mod_matrix" in msg:
                    self.set_mod_matrix(msg["mod_matrix"])
                if "input_matrix" in msg:
                    self.set_input_matrix(msg["input_matrix"])
                if "mix_mode" in msg:
                    self.set_mix_mode(msg["mix_mode"])

    def run(self) -> None:
        self.running = True
        listener = threading.Thread(target=self.udp_listener, daemon=True)
        listener.start()

        if AUDIO_BACKEND == "sounddevice" and sd is not None:
            def callback(outdata, frames, _time_info, _status):
                block = self.compute_frame(frames)
                outdata[:] = block.reshape(-1, 1)

            with sd.OutputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                dtype="float32",
                callback=callback,
            ):
                while self.running:
                    time.sleep(0.1)
            return

        while self.running:
            block = self.compute_frame(BLOCKSIZE)
            pcm = (block * 32767.0).astype(np.int16)
            if sa is not None:
                play = sa.play_buffer(pcm.tobytes(), 1, 2, SAMPLE_RATE)
                play.wait_done()
            else:
                time.sleep(BLOCKSIZE / SAMPLE_RATE)


if __name__ == "__main__":
    engine = HyperCubeAudio(num_ops=12)
    engine.run()
