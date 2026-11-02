#include <cmath>
#include <cstdint>
#include <initializer_list>

#include "rand.h"


#define SIZE   256 ///< 必须为偶数
#define WIDTH  SIZE
#define HEIGHT SIZE
#define N()    (WIDTH * HEIGHT)


#define x(i)      ((i) % (WIDTH))
#define y(i)      ((i) / (WIDTH))
#define idx(x, y) ((x) + (y) * (WIDTH))

struct Ising {
  constexpr static double N = N();
  double                  K, h = 0.;
  double                  m = 0.;

private:
  int8_t state[N()];

  uint64_t seed;
  uint64_t sweeps = 0;
  double   lut[10];

public:
  Ising(double K, double h, uint64_t seed) : K(K), h(h), seed(seed) {
    for (int32_t i = 0; i < N(); i++) {
      int8_t s       = randd(this->seed, i) > 0.5 ? 1 : -1;
      this->state[i] = s;
      this->m += s;
    }
    this->m /= this->N;
#define lu_idx(s, nbsum) (((s) + 1) / 2 + ((((nbsum) + 4) / 2) << 1))
    for (int8_t ss : {-1, 1}) {
      for (int8_t nn : {-4, -2, 0, 2, 4}) {
        lut[lu_idx(ss, nn)] = exp(-2. * ss * (h + K * nn));
      }
    }
  }

  void set_ones() {
    for (int32_t i = 0; i < N(); i++)
      this->state[i] = 1;
    this->m = 1.;
  }

  void sweep() {
    this->sweeps++;
    for (int32_t c = 0; c <= 1; c++) {
      int32_t delta_m = 0.;
#pragma omp parallel for reduction(+ : delta_m) schedule(static)
      for (int32_t y = 0; y < HEIGHT; y++) {
        int32_t bg = c ^ (y & 1);
        for (int32_t x = bg; x < WIDTH; x += 2) {

          int8_t nbsum = 0;
          for (int8_t dir = 0; dir < 4; dir++) {
            int32_t arg = dir & 0b01, len = (dir & 0b10) - 1;
            int32_t dx = arg * len, dy = (1 - arg) * len;
            int32_t xx = (x + dx + WIDTH) % WIDTH;
            int32_t yy = (y + dy + HEIGHT) % HEIGHT;
            nbsum += this->state[idx(xx, yy)];
          }

          uint32_t idx   = idx(x, y);
          int8_t   s     = this->state[idx];
          double   omega = lookup(s, nbsum);
          double   r     = randd(this->seed, sweeps * N() + idx);
          if (omega > r) {
            this->state[idx] *= -1;
            delta_m += 2 * this->state[idx];
          }
        }
      }
      this->m += (double)delta_m / this->N;
    }
  }

private:
  inline double lookup(int8_t s, int8_t nbsum) const {
    return lut[lu_idx(s, nbsum)];
#undef lu_idx
  }
};


int main() {
  Ising ising(0.49, 0., 114514);
  for (uint32_t _ = 0; _ < 10000; _++)
    ising.sweep();
}
