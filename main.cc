#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "rand.h"


struct Ising {
  uint32_t L;
  double   K;
  double   m = 0., e = 0.;

  int8_t  *states = nullptr;
  uint64_t seed;

public:
  Ising(uint32_t L, double K, uint64_t seed) : L(L), K(K), seed(seed) {
    this->states = (int8_t *)std::malloc(L * L);
  }

  ~Ising() { std::free(this->states); }

  int8_t nbsum(uint32_t idx) {
    int8_t nbsum = 0;
    for (int8_t dir = 0; dir < 4; dir++) {
      int32_t arg = dir & 0b01, len = (dir & 0b10) - 1;
      int32_t dx = arg * len, dy = (1 - arg) * len;
      int32_t xx = (idx + dx + L) % L;
      int32_t yy = (idx / L + dy + L) % L;
      nbsum += this->states[xx + yy * L];
    }
    return nbsum;
  }

  void set_ones() {
    for (uint32_t i = 0; i < L * L; i++)
      this->states[i] = 1;
    this->m = 1.;
    this->e = -2. * K;
  }

  void set_random() {
    double M = 0.;
    for (uint32_t i = 0; i < L * L; i++) {
      double s        = randd(seed) >= 0.5 ? 1 : -1;
      this->states[i] = s;
      M += s;
    }
    double E = 0.;
    for (uint32_t i = 0; i < L * L; i++) {
      E += -.5 * K * this->states[i] * nbsum(i);
    }
    this->m = M / (L * L);
    this->e = E / (L * L);
  }

  void _cluster_flip(uint32_t now) {
    int8_t s = this->states[now];

    // 找到所有邻居
    uint32_t nb[4];
    for (uint32_t dir = 0; dir < 4; dir++) {
      int32_t arg = dir & 0b01, len = (dir & 0b10) - 1;
      int32_t dx = arg * len, dy = (1 - arg) * len;
      int32_t xx = (now + dx + L) % L;
      int32_t yy = (now / L + dy + L) % L;
      nb[dir]    = xx + yy * L;
    }

    // 翻转当前格点自旋
    this->states[now] *= -1;

    int8_t nbsum = 0;
    for (uint32_t i = 0; i < 4; i++)
      nbsum += this->states[nb[i]];
    // 计算由于这次翻转导致的磁化强度与能量变化
    this->m += -2. * s / (L * L);
    // 每个键具有 -K s_i s_j 能量
    // ΔE = -K (-s) nbsum - (-K) s nbsum = 2 K s nbsum
    this->e += 2. * K * s * nbsum / (L * L);

    for (int8_t i = 0; i < 4; i++) {
      // 向四周进行深度优先遍历
      uint32_t next = nb[i];
      // 仅当邻居与自己自旋相同时才划入同一集团.
      // 因自己的自旋已被翻转, 进入邻居后不会再返回自己,
      //  避免陷入循环.
      if (s == this->states[next]) {
        double p = 1. - exp(-2. * K);
        if (p > randd(seed))
          _cluster_flip(next);
      }
    }
  }

  void sweep() {
    uint32_t idx = L * L * randd(seed);
    _cluster_flip(idx);
  }
};


int main() { return 0; }
