"""
Operator Module (CODEX 4)

Implements the Vedic sutra operator stack and GRVQ/MSTVQ/R4 operators.

All operators share a common interface:
    apply(state, context) -> state'

Requirements:
- Composable (pipeline)
- Supports exact arithmetic mode
- Logs action in structured trace (for replay + proofs)
"""

from .base import (
    Operator,
    OperatorContext,
    OperatorCategory,
    OperatorTrace,
    CompositeOperator,
    IdentityOperator,
)

__all__ = [
    'Operator',
    'OperatorContext',
    'OperatorCategory',
    'OperatorTrace',
    'CompositeOperator',
    'IdentityOperator',
]
