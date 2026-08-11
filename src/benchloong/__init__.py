"""2024 年全国大学生数学建模竞赛 A 题“板凳龙”共享模型库。"""

from .config import (
    N_BENCHES,
    N_HANDLES,
    HANDLE_GAPS,
    BOARD_OVERHANG,
    HALF_WIDTH,
    THETA0,
)
from .geometry import (
    G,
    G_inv,
    IncomingSpiral,
    TurnPath,
)
from .chain import (
    ChainError,
    chain_spiral,
    rear_handle_s,
    chain_arc_params,
    chain_speeds,
    bench_rectangles,
    any_collision,
)

__all__ = [
    "N_BENCHES",
    "N_HANDLES",
    "HANDLE_GAPS",
    "BOARD_OVERHANG",
    "HALF_WIDTH",
    "THETA0",
    "G",
    "G_inv",
    "IncomingSpiral",
    "TurnPath",
    "ChainError",
    "chain_spiral",
    "rear_handle_s",
    "chain_arc_params",
    "chain_speeds",
    "bench_rectangles",
    "any_collision",
]
