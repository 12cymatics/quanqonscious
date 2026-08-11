"""Tesseract working memory: project hidden states into a 16-slot Z₂⁴ vector."""
from __future__ import annotations

from .slot_map import SLOT_NAMES, slot_index_for, vertex_for_name
from .tesseract_wm import TesseractWM

__all__ = ["TesseractWM", "SLOT_NAMES", "slot_index_for", "vertex_for_name"]
