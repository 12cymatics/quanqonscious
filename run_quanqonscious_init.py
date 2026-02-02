#!/usr/bin/env python3
"""
One-click launcher for the QuanQonscious hybrid quantum-classical simulator.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, List, Optional

from hc_ipc import HcIpcClient


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    env: Dict[str, str]
    process: Optional[subprocess.Popen] = None
    thread: Optional[threading.Thread] = None


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QuanQonscious One-Click Launcher")
        self.geometry("1100x720")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.processes: Dict[str, ManagedProcess] = {}
        self.ipc = HcIpcClient()

        self._build_ui()
        self.after(100, self._flush_logs)

    def _build_ui(self) -> None:
        header = ttk.Label(self, text="QuanQonscious Hybrid Simulator", font=("Segoe UI", 18, "bold"))
        header.pack(pady=10)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=10)

        ttk.Button(controls, text="Start Audio Engine", command=self.start_audio).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(controls, text="Stop Audio Engine", command=self.stop_audio).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(controls, text="Start Web Server", command=self.start_web).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(controls, text="Run Hybrid Simulator", command=self.run_hybrid).grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(controls, text="Run H₂ Simulation", command=self.run_h2).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(controls, text="Run Full 30-Sutra Cymatics", command=self.run_cymatics).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(controls, text="Run Full Pipeline", command=self.run_full_pipeline).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(controls, text="Stop All", command=self.stop_all).grid(row=1, column=3, padx=5, pady=5)

        self.log_text = tk.Text(self, wrap=tk.NONE, height=28)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _flush_logs(self) -> None:
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.insert(tk.END, msg)
            self.log_text.see(tk.END)
        self.after(100, self._flush_logs)

    def _start_process(self, name: str, command: List[str], enable_audio: bool = False) -> None:
        if name in self.processes and self.processes[name].process is not None:
            self._log(f"[{name}] already running.\n")
            return

        env = os.environ.copy()
        if enable_audio:
            env["QUANQONSCIOUS_AUDIO"] = "1"

        self._log(f"[{name}] starting: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        def reader() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                self._log(f"[{name}] {line}")

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        self.processes[name] = ManagedProcess(name=name, command=command, env=env, process=process, thread=thread)

    def _stop_process(self, name: str) -> None:
        managed = self.processes.get(name)
        if not managed or managed.process is None:
            return
        self._log(f"[{name}] stopping.\n")
        managed.process.terminate()
        managed.process = None

    def start_audio(self) -> None:
        self._start_process("audio", [sys.executable, "audio_output.py"])
        self.ipc.start()

    def stop_audio(self) -> None:
        self.ipc.stop()
        self._stop_process("audio")

    def start_web(self) -> None:
        self._start_process("web", [sys.executable, "web_server.py"])

    def run_hybrid(self) -> None:
        self._start_process(
            "hybrid",
            [sys.executable, "hybrid_simulator.py", "1.0", "--run-all-modes", "--enable-audio"],
            enable_audio=True,
        )

    def run_h2(self) -> None:
        self._start_process(
            "h2",
            [sys.executable, "run_h2_grvq_simulation.py"],
            enable_audio=True,
        )

    def run_cymatics(self) -> None:
        self._start_process(
            "cymatics",
            [sys.executable, "full_30_sutra_cymatic_engine.py"],
            enable_audio=True,
        )

    def run_full_pipeline(self) -> None:
        self.start_audio()
        self.start_web()
        self.run_hybrid()
        self.run_h2()
        self.run_cymatics()

    def stop_all(self) -> None:
        for name in list(self.processes.keys()):
            self._stop_process(name)


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
