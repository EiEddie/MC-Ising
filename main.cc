#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <tuple>

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

  /**
   * @brief 热化系统
   *
   * 使系统达到热平衡状态.
   *
   * 使用方差停机逻辑, 当 m (或 |m|) 标准差 σ 小于 eps 时认为达到平衡.
   * 前 drop 次 sweep 将会被丢弃;
   * 必须指定 drop 并至少使 drop>1, 否则第一次即会返回.
   * 最多将会运行 drop+max 次, 超出即停止.
   *
   * 将返回 (sweeps, ⟨m⟩, σ); 当外场 h=0 时, 将返回 (sweeps, ⟨|m|⟩, σ).
   * 当因超出 max 而停机, 返回 (drop+max, nan, nan).
   */
  std::tuple<uint32_t, double, double> equilibrate(double eps, uint32_t max, uint32_t drop = 300) {
    constexpr uint32_t CAPA = 30;
    constexpr uint32_t STEP = 10;

    double m_s[CAPA]{0}, m_sq_s[CAPA]{0};
    double m_sum = 0., m_sq_sum = 0.;
    // cnt 从 1 开始
    // 否则 drop=0 时, 第一次便会进入判断逻辑
    // 此时 len=cnt/STEP=i=0, 计算出现问题
    for (uint32_t cnt = 1; cnt <= max; cnt++) {
      this->sweep();
      if (cnt % STEP != 0)
        continue;

      uint32_t i = cnt / STEP % CAPA;
      m_sum -= m_s[i];
      m_sq_sum -= m_sq_s[i];
      // 当外场 h 为 0 时, 使用 |m| 而非 m.
      // 因为此时存在一个整体翻转的对称性.
      m_s[i]    = this->h == 0. ? std::abs(this->m) : this->m;
      m_sq_s[i] = m_s[i] * m_s[i];
      m_sum += m_s[i];
      m_sq_sum += m_sq_s[i];
#define MIN(x, y) ((x) > (y) ? (y) : (x))
      double len = MIN(CAPA, cnt / STEP);
#undef MIN
      double m_avg = m_sum / len, m_sq_avg = m_sq_sum / len;
      double var = m_sq_avg - m_avg * m_avg;
#define MAX(x, y) ((x) > (y) ? (x) : (y))
      // 当 m 很平缓时, 可能出现浮点误差导致的负方差.
      var = MAX(0., var);
      // std::cout << "var:" << var << std::endl;
#undef MAX
      if (cnt > drop && var < eps * eps)
        return std::make_tuple(drop + cnt, m_avg, std::sqrt(var));
    }
    return std::make_tuple(drop + max, NAN, NAN);
  }

private:
  inline double lookup(int8_t s, int8_t nbsum) const {
    return lut[lu_idx(s, nbsum)];
#undef lu_idx
  }
};


int main() {
  Ising ising(0.49, 0., 114514);
  auto [sweeps, m_avg, m_std] = ising.equilibrate(5e-3, 5e4, 300);
  std::printf("sweeps=%u; ⟨m⟩=%.4f; σ=%.4f", sweeps, m_avg, m_std);
}
