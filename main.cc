#include <cmath>
#include <cstdint>
#include <random>


std::uniform_real_distribution<double> uniform{0., 1.};


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

private:
  int8_t state[N()];

  std::mt19937_64 rng;
  double          rands[N()];

public:
  Ising(double K, double h, uint64_t seed) : K(K), h(h), rng(seed) {
    for (int32_t i = 0; i < N(); i++)
      this->state[i] = (rng() >> 63) ? 1 : -1;
  }

  void sweep() {
    for (uint32_t i = 0; i < N(); i++)
      rands[i] = uniform(this->rng);

    for (int32_t c = 0; c <= 1; c++) {
#pragma omp parallel for schedule(static)
      for (int32_t y = 0; y < HEIGHT; y++) {
        int32_t bg = c ^ (y & 1);
        for (int32_t x = bg; x < WIDTH; x += 2) {
          int8_t nbsum = 0;
          for (int8_t dir = 0; dir < 4; dir++) {
            int32_t dx = (dir & 0b01) * 2 - 1, dy = (dir & 0b10) - 1;
            int32_t xx = (x + dx + WIDTH) % WIDTH;
            int32_t yy = (y + dy + HEIGHT) % HEIGHT;
            nbsum += this->state[idx(xx, yy)];
          }

          uint32_t idx   = idx(x, y);
          int8_t   s     = this->state[idx];
          double   omega = exp(-2. * s * (h + K * nbsum));
          if (omega > rands[idx])
            this->state[idx] *= -1;
        }
      }
    }
  }
};


int main() {
  Ising ising(0.49, 0., 114514);
  for (uint32_t _ = 0; _ < 10000; _++)
    ising.sweep();
}
