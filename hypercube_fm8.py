#!/usr/bin/env python3
"""
FM8-style HyperCube control surface for sutra-driven frequency shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import importlib.util

if importlib.util.find_spec("numpy") is None:
    raise ImportError("numpy is required for hypercube_fm8.py")
import numpy as np


@dataclass
class OperatorState:
    base_freq: float
    ratio: float = 1.0
    detune: float = 0.0
    level: float = 0.5
    feedback: float = 0.0


@dataclass
class SutraMapping:
    operator_indices: List[int]
    freq_scale: float = 0.01
    level_scale: float = 0.002
    ratio_scale: float = 0.001
    detune_scale: float = 0.5
    offset: float = 0.0


class HyperCubeFM8:
    def __init__(self, num_ops: int = 8, base_frequency: float = 432.0) -> None:
        self.num_ops = max(1, int(num_ops))
        self.base_frequency = float(base_frequency)
        self.operators: List[OperatorState] = [
            OperatorState(base_freq=self.base_frequency, ratio=1.0 + i * 0.05, level=0.2)
            for i in range(self.num_ops)
        ]
        self.mod_matrix = np.zeros((self.num_ops, self.num_ops), dtype=float)
        self.input_matrix = np.zeros((self.num_ops, 1), dtype=float)
        self.mix_mode = "concurrent"
        self.sutra_mappings: Dict[str, SutraMapping] = {}

    def set_input_matrix(self, matrix: Sequence[Sequence[float]]) -> None:
        if len(matrix) == 0:
            return
        rows = []
        for row in matrix:
            if isinstance(row, (list, tuple, np.ndarray)):
                rows.append([float(v) for v in row])
            else:
                rows.append([float(row)])
        self.input_matrix = np.array(rows, dtype=float)

    def set_modulation_matrix(self, matrix: Sequence[Sequence[float]]) -> None:
        mat = np.array(matrix, dtype=float)
        if mat.shape != (self.num_ops, self.num_ops):
            raise ValueError("modulation matrix must be shape (num_ops, num_ops)")
        self.mod_matrix = mat

    def set_mix_mode(self, mode: str) -> None:
        if mode not in ("concurrent", "parallel", "serial"):
            raise ValueError("mix_mode must be concurrent|parallel|serial")
        self.mix_mode = mode

    def set_operator(
        self,
        index: int,
        base_freq: Optional[float] = None,
        ratio: Optional[float] = None,
        detune: Optional[float] = None,
        level: Optional[float] = None,
        feedback: Optional[float] = None,
    ) -> None:
        if index < 0 or index >= self.num_ops:
            raise IndexError("operator index out of range")
        op = self.operators[index]
        if base_freq is not None:
            op.base_freq = float(base_freq)
        if ratio is not None:
            op.ratio = float(ratio)
        if detune is not None:
            op.detune = float(detune)
        if level is not None:
            op.level = float(level)
        if feedback is not None:
            op.feedback = float(feedback)

    def add_sutra_mapping(
        self,
        sutra_name: str,
        operator_indices: Optional[Iterable[int]] = None,
        freq_scale: float = 0.01,
        level_scale: float = 0.002,
        ratio_scale: float = 0.001,
        detune_scale: float = 0.5,
        offset: float = 0.0,
    ) -> None:
        indices = list(operator_indices) if operator_indices is not None else list(range(self.num_ops))
        self.sutra_mappings[sutra_name] = SutraMapping(
            operator_indices=indices,
            freq_scale=freq_scale,
            level_scale=level_scale,
            ratio_scale=ratio_scale,
            detune_scale=detune_scale,
            offset=offset,
        )

    def _normalize_value(self, value: float) -> float:
        return float(np.tanh(value))

    def apply_sutra_to_operators(self, sutra_name: str, values: Sequence[float]) -> None:
        if sutra_name not in self.sutra_mappings:
            self.add_sutra_mapping(sutra_name)
        mapping = self.sutra_mappings[sutra_name]
        aggregate = float(np.mean(values)) if len(values) else 0.0
        normalized = self._normalize_value(aggregate)

        for idx in mapping.operator_indices:
            if idx < 0 or idx >= self.num_ops:
                continue
            op = self.operators[idx]
            op.base_freq = max(20.0, op.base_freq * (1.0 + mapping.freq_scale * normalized) + mapping.offset)
            op.level = max(0.0, min(1.0, op.level + mapping.level_scale * normalized))
            op.ratio = max(0.1, op.ratio + mapping.ratio_scale * normalized)
            op.detune = op.detune + mapping.detune_scale * normalized

    def operator_frequencies(self) -> List[float]:
        freqs = []
        for idx, op in enumerate(self.operators):
            base = op.base_freq * op.ratio + op.detune
            if idx < self.input_matrix.shape[0]:
                base += float(np.mean(self.input_matrix[idx]))
            freqs.append(max(20.0, min(18000.0, base)))
        return freqs

    def operator_levels(self) -> List[float]:
        return [max(0.0, min(1.0, op.level)) for op in self.operators]

    def as_update_payload(self) -> dict:
        return {
            "base_ops": self.operator_frequencies(),
            "levels": self.operator_levels(),
            "mod_matrix": self.mod_matrix.tolist(),
            "input_matrix": self.input_matrix.tolist(),
            "mix_mode": self.mix_mode,
        }
