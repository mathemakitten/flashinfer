// Experimental GB300 BF16 overlap control. Read once per process, before graph capture.
#pragma once
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace flashinfer_nt4 {
inline int max_blocks() {
#ifdef FLASHINFER_NT4_BLOCK_CAP
  static const int value = [] {
    const char* text = std::getenv("NT4_CUTLASS_MAX_BLOCKS");
    if (!text) return 0;
    char* end = nullptr;
    long n = std::strtol(text, &end, 10);
    if (end == text || *end || n < 0 || n > 4096 || n % 16 != 0)
      throw std::invalid_argument("NT4_CUTLASS_MAX_BLOCKS must be 0 or a multiple of 16 <= 4096");
    return static_cast<int>(n);
  }();
  return value;
#else
  return 0;
#endif
}
inline int limit(int natural) {
  int cap = max_blocks();
  return cap ? std::min(natural, cap) : natural;
}
}  // namespace flashinfer_nt4
