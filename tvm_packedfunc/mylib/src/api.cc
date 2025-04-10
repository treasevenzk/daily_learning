#include "mylib/api.h"
#include "mylib/runtime/registry.h"
#include <numeric>
#include <string>

namespace mylib {

int Add(int a, int b) {
  return a + b;
}

std::string Repeat(const std::string& input, int n) {
  std::string result;
  for (int i = 0; i < n; ++i) {
    result += input;
  }
  return result;
}

int SumVector(const std::vector<int>& vec) {
  return std::accumulate(vec.begin(), vec.end(), 0);
}

// 注册这些函数到全局注册表
// 使用全局工厂函数而不是宏
namespace {
  
// 静态变量的初始化会在程序开始时执行
// 这些函数注册会自动完成
static auto _ = runtime::RegisterFunction("mylib.Add")
                .set_body(runtime::PackFuncTyped(Add));

static auto __ = runtime::RegisterFunction("mylib.Repeat")
                 .set_body(runtime::PackFuncTyped(Repeat));

static auto ___ = runtime::RegisterFunction("mylib.SumVector")
                  .set_body(runtime::PackFuncTyped(SumVector));

} // namespace

}  // namespace mylib