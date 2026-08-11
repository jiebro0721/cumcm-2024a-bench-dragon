"""问题常量：板凳几何尺寸、初始条件。"""

import numpy as np

# ---- 板凳几何（单位：m） ----
HEAD_BOARD_LEN = 3.41        # 龙头板长
BODY_BOARD_LEN = 2.20        # 龙身 / 龙尾板长
HOLE_OFFSET = 0.275          # 孔中心到最近板头
BOARD_WIDTH = 0.30           # 板宽

N_BENCHES = 223              # 板凳总数（龙头 1 + 龙身 221 + 龙尾 1）
N_HANDLES = N_BENCHES + 1    # 把手点数：P0..P223（224 个）

# l_i = 第 i 条板凳前、后把手之间的距离（i = 0..222）
# 龙头：341 - 2*27.5 = 286 cm；其余：220 - 2*27.5 = 165 cm
_gaps = np.full(N_BENCHES, BODY_BOARD_LEN - 2.0 * HOLE_OFFSET)
_gaps[0] = HEAD_BOARD_LEN - 2.0 * HOLE_OFFSET
HANDLE_GAPS = _gaps

BOARD_OVERHANG = HOLE_OFFSET        # 板头伸出孔外的长度 d_i
HALF_WIDTH = BOARD_WIDTH / 2.0      # 板凳半宽 omega_i

# ---- 初始条件 ----
# 龙头初始位于第 16 圈 A 点（正 x 轴、最外端），theta_0(0) = 16 * 2*pi = 32*pi
THETA0 = 32.0 * np.pi
