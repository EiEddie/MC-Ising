import numpy as np
from numpy.typing import NDArray
from numba import njit
import matplotlib.pyplot as plt

SpinArray = NDArray[np.int8]


class Ising:
    def __init__(self, size: int, K: float, h: float, seed=None):
        # K, h 都是无量纲的
        # K = βJ; h = βh
        self.K = K
        self.h = h

        self._rng = np.random.default_rng(seed)

        self.N = np.float64(size**2)
        # size 必须为偶数
        self._shape = (size, size//2)
        self._states: list[SpinArray] = [np.ones(self._shape, dtype=np.int8),
                                         np.ones(self._shape, dtype=np.int8)]


    def _reset_rand(self):
        for i in (0,1):
            random_bits = self._rng.integers(0, 2, size=self._shape)
            self._states[i][:] = np.where(random_bits == 0, -1, 1).astype(np.int8)


    def _reset_uniform(self):
        for i in (0,1):
            self._states[i].fill(1)


    def reset(self, random=True):
        if random:
            self._reset_rand()
        else:
            self._reset_uniform()
        return self


    def sweep(self):
        _sweep(self._states, self.K, self.h, self._rng)


    def m(self) -> np.float64:
        return np.sum(self._states).astype(np.float64) / self.N


@njit
def _nbsum(grid: SpinArray, dir) -> SpinArray:
    n, m = grid.shape
    res = np.empty_like(grid)

    for r in range(n):
        rm = r - 1 if r > 0 else n - 1
        rp = r + 1 if r + 1 < n else 0
        for b in range(m):
            hb = b - dir if r % 2 == 0 else b + dir
            if hb < 0:
                hb += m
            elif hb >= m:
                hb -= m

            res[r, b] = (grid[r, b] + grid[rm, b]
                         + grid[rp, b] + grid[r, hb])
    return res


@njit
def _sweep(states: list[SpinArray], K: float, h: float,
           rng=np.random.default_rng()):
    for i in (0,1):
        tmp = states[i].copy().astype(np.float64)
        # 计算邻居和
        # 应当使用另一个格子来计算, 即 `not i === 1-i`
        # states[0] 为黑格, 应加右侧, dir=1; 反之 [1]: dir=-1
        nbsum = _nbsum(states[1-i], dir=-2*i+1).astype(np.float64)
        tmp = np.exp(-2.*tmp*(h + K*nbsum)) - rng.random(tmp.shape)
        # 更新格子状态
        states[i] *= np.where(tmp > 0, np.int8(-1), np.int8(1))


size = 256
K = 0.49

if __name__ == '__main__':
    seed = np.random.randint(0, 2**32, dtype=np.uint32)
    print(f"seed={seed}")
    ising = Ising(size, K, 0., seed=seed).reset()

    m_avg = float('nan')
    m_sq_avg = float('nan')
    m_var = float('nan')

    # 至少运行的 sweep 轮数: cap*step
    cap = 30; step = 10
    m_s = np.zeros(cap, dtype=np.float64)
    m_sq_s = np.zeros_like(m_s)
    idx = 0
    while True:
        idx += 1
        if idx % step == 0:
            i = idx // step % cap
            m = m_s[i] = ising.m()
            m_sq_s[i] = m**2

            N = np.minimum(cap, idx // step).astype(np.float64)
            avg = np.sum(m_s) / N
            sq_avg = np.sum(m_sq_s) / N
            var = sq_avg - avg**2
            if idx > 300 and var < 0.005**2:
                m_avg = avg
                m_sq_avg = sq_avg
                m_var = var
                break
        ising.sweep()

    print(f"sweep={idx}; ⟨m⟩={m_avg:.4f}; "
          f"⟨m²⟩={m_sq_avg:.4f}; σ={np.sqrt(m_var):.5f}")
