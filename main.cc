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
    // 初始化查找表
    // 自旋 s 共 2 个态: ±1;
    // 邻居和 nbsum 共 5 个态: ±4, ±2, 0.
    for (int8_t ss : {-1, 1}) {
      for (int8_t nn : {-4, -2, 0, 2, 4}) {
        lut[lu_idx(ss, nn)] = exp(-2. * ss * (h + K * nn));
      }
    }
  }

  /**
   * @brief 将网格重置为全正自旋 (s=1)
   */
  void set_ones() {
    for (int32_t i = 0; i < N(); i++)
      this->state[i] = 1;
    this->m = 1.;
  }

  void sweep() {
    this->sweeps++;
    // 分黑白棋盘格处理
    // 黑格的邻居必为白格,
    // 同色之间无数据竞争, 但异色存在,
    // 因此只能并行计算网格的一半.
    for (int32_t c = 0; c <= 1; c++) {
      int32_t delta_m = 0.;
#pragma omp parallel for reduction(+ : delta_m) schedule(static)
      for (int32_t y = 0; y < HEIGHT; y++) {
        // 换色时交错开始迭代
        int32_t bg = c ^ (y & 1);
        for (int32_t x = bg; x < WIDTH; x += 2) {

          // 计算邻居和
          int8_t nbsum = 0;
          for (int8_t dir = 0; dir < 4; dir++) {
            // arg 表示方向:
            // dir=0, 1-dir=1 表示横向 (x); dir=1, 1-dir=0 表示纵向 (y).
            // len 表示步长:
            // 长度恒为 1; 方向正负不定.
            int32_t arg = dir & 0b01, len = (dir & 0b10) - 1;
            int32_t dx = arg * len, dy = (1 - arg) * len;
            // 周期性边界条件
            int32_t xx = (x + dx + WIDTH) % WIDTH;
            int32_t yy = (y + dy + HEIGHT) % HEIGHT;
            nbsum += this->state[idx(xx, yy)];
          }

          // 翻转判断
          uint32_t idx   = idx(x, y);
          int8_t   s     = this->state[idx];
          double   omega = lookup(s, nbsum);
          // 使用无状态的随机数生成器
          // 保证同种子, 同 sweep 次数, 同格点给出的随机数一致,
          // 不受线程调度影响.
          double   r     = randd(this->seed, this->sweeps * N() + idx);
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
