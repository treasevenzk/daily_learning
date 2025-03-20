#include "mylib/runtime/packed_func.h"
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

namespace mylib {
namespace runtime {

// 实现TVMArgs的Get方法
template<>
int TVMArgs::Get<int>(int i) const {
  if (i >= num_args_ || type_codes_[i] != TVMPODValue_::kInt) {
    throw std::invalid_argument("TypeError: Argument " + std::to_string(i) + " is not an int");
  }
  return static_cast<int>(values_[i].value_.v_int64);
}

template<>
int64_t TVMArgs::Get<int64_t>(int i) const {
  if (i >= num_args_ || type_codes_[i] != TVMPODValue_::kInt) {
    throw std::invalid_argument("TypeError: Argument " + std::to_string(i) + " is not an int64");
  }
  return values_[i].value_.v_int64;
}

template<>
double TVMArgs::Get<double>(int i) const {
  if (i >= num_args_ || type_codes_[i] != TVMPODValue_::kFloat) {
    throw std::invalid_argument("TypeError: Argument " + std::to_string(i) + " is not a double");
  }
  return values_[i].value_.v_float64;
}

template<>
std::string TVMArgs::Get<std::string>(int i) const {
  if (i >= num_args_ || type_codes_[i] != TVMPODValue_::kStr) {
    throw std::invalid_argument("TypeError: Argument " + std::to_string(i) + " is not a string");
  }
  return std::string(values_[i].value_.v_str);
}

// PackedFunc的operator()实现
// 这里只提供一个简单的实现，实际应用中需要更复杂的处理
template<typename... Args>
TVMRetValue PackedFunc::operator()(Args&&... args) const {
  // 简化实现，仅用于演示
  TVMRetValue rv;
  // 实际应用中应该收集参数并调用CallPacked
  return rv;
}

// 不要使用模板特化语法，这会导致编译错误

}  // namespace runtime
}  // namespace mylib