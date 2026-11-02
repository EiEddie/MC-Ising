#ifndef _H_RAND
#define _H_RAND

#include <cstdint>


#define map_rand_to_unif(r) ((double)((r) >> 11) * 0x1.0p-53)

static inline uint64_t _splitmix_final(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static inline double randd(uint64_t seed, uint64_t counter) {
  constexpr uint64_t golden = 0x9e3779b97f4a7c15ULL;
  uint64_t           bits   = _splitmix_final(seed + counter * golden);
  return map_rand_to_unif(bits);
}

#endif
