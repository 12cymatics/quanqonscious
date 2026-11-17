#pragma once
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace faraday_native {
inline std::size_t memoryProduct(std::size_t a,std::size_t b) {
  if(b && a>std::numeric_limits<std::size_t>::max()/b)
    throw std::length_error("Native working-array size overflows; no allocation performed");
  return a*b;
}
inline std::size_t memorySum(std::initializer_list<std::size_t> parts) {
  std::size_t n=0;
  for(auto part:parts){
    if(part>std::numeric_limits<std::size_t>::max()-n)
      throw std::length_error("Native working-array size overflows; no allocation performed");
    n+=part;
  }
  return n;
}
inline void requireMemory(std::size_t need,std::size_t budget) {
  if(need>budget)throw std::length_error("Native coupling working arrays exceed resourceBudgetBytes; no grain, grid or force-law substitution performed");
}
} // namespace faraday_native
