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
      double s        = randd(seed);
      this->states[i] = s;
      M += s;
    }
    double E = 0.;
    for (uint32_t i = 0; i < L; i++) {
      E += -.5 * K * this->states[i] * nbsum(i);
    }
    this->m = M / (L * L);
    this->e = E / (L * L);
  }

  void _sweep(uint32_t now) {
    int8_t s = this->states[now];
    this->states[now] *= -1;

    for (int8_t dir = 0; dir < 4; dir++) {
      int32_t  arg = dir & 0b01, len = (dir & 0b10) - 1;
      int32_t  dx = arg * len, dy = (1 - arg) * len;
      int32_t  xx   = (now + dx + L) % L;
      int32_t  yy   = (now / L + dy + L) % L;
      uint32_t next = xx + yy * L;

      if (s != this->states[next])
        continue;
      double p = 1. - exp(-K);
      if (p < randd(seed))
        _sweep(next);
    }
  }

  void sweep() {
    uint32_t idx = L * L * randd(seed);
    _sweep(idx);
  }
};
