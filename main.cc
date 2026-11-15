#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "rand.h"


struct Ising {
  uint32_t L;
  double   K;
  double   m = 0., e = 0.;

  int8_t  *states = nullptr;
  uint64_t seed, counter = 0;

private:
  void nb(uint32_t buf[4], uint32_t idx) {
    for (int8_t dir = 0; dir < 4; dir++) {
      int32_t arg = dir & 0b01, len = (dir & 0b10) - 1;
      int32_t dx = arg * len, dy = (1 - arg) * len;
      int32_t xx = (idx + dx + L) % L;
      int32_t yy = (idx / L + dy + L) % L;
      buf[dir]   = xx + yy * L;
    }
  }

  void flip(uint32_t idx) {
    int8_t s = this->states[idx];
    // 翻转当前格点自旋
    this->states[idx] *= -1;

    uint32_t nb[4];
    int8_t   nbsum = 0;
    this->nb(nb, idx);
    for (uint32_t i = 0; i < 4; i++)
      nbsum += this->states[nb[i]];

    // 计算由于这次翻转导致的磁化强度与能量变化
    this->m += -2. * s / (L * L);
    // 每个键具有 -K s_i s_j 能量
    // ΔE = -K (-s) nbsum - (-K) s nbsum = 2 K s nbsum
    this->e += 2. * K * s * nbsum / (L * L);
  }

  int8_t nbsum(uint32_t idx) {
    uint32_t nb[4];
    this->nb(nb, idx);
    int8_t nbsum = 0;
    for (uint32_t i = 0; i < 4; i++)
      nbsum += this->states[nb[i]];
    return nbsum;
  }

public:
  Ising(uint32_t L, double K, uint64_t seed) : L(L), K(K), seed(seed) {
    this->states = (int8_t *)std::malloc(L * L);
  }

  Ising(const Ising &self) : Ising(self.L, self.K, self.seed) {}

  ~Ising() { std::free(this->states); }

  void set_ones() {
    for (uint32_t i = 0; i < L * L; i++)
      this->states[i] = 1;
    this->m = 1.;
    this->e = -2. * K;
  }

  void set_random() {
    double M = 0.;
    for (uint32_t i = 0; i < L * L; i++) {
      double s        = randd(seed, this->counter++) >= 0.5 ? 1 : -1;
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

  void update() {
    std::vector<uint32_t> cluster{};
    cluster.push_back(L * L * randd(seed));
    // 每个格点入栈前必须翻转并计算磁化强度与能量的变化量
    this->flip(cluster.back());

    while (!cluster.empty()) {
      uint32_t now = cluster.back();
      cluster.pop_back();

      uint32_t nb[4];
      this->nb(nb, now);
      for (uint32_t i = 0; i < 4; i++) {
        uint32_t next = nb[i];
        double   p    = 1. - exp(-2. * K);
        // 必须为同一自旋;
        //   且成键概率被接受.
        // 当前自旋 states[now] 已被翻转, 因此逻辑取反.
        if (states[next] != -states[now] || p < randd(seed, this->counter++))
          continue;
        this->flip(next);
        cluster.push_back(next);
      }
    }
  }

  double energy() const { return this->e; }

  double magnetization() const { return std::abs(this->m); }
};


int main() { return 0; }
