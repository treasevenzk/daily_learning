#include <cassert>
#include <iostream>
#include <string>

#include "packed_fun.h"
#include "registry.h"

using namespace packedfun;

// Test functions
PackedValue TestAdd(const std::vector<PackedValue>& args) {
  return PackedValue(args[0].AsInt() + args[1].AsInt());
}

PackedValue TestMultiply(const std::vector<PackedValue>& args) {
  return PackedValue(args[0].AsInt() * args[1].AsInt());
}

int main() {
  // Register test functions
  Registry::Global()->Register("TestAdd", TestAdd);
  Registry::Global()->Register("TestMultiply", TestMultiply);
  
  // Test function retrieval
  assert(Registry::Global()->Contains("TestAdd"));
  assert(Registry::Global()->Contains("TestMultiply"));
  assert(!Registry::Global()->Contains("NonExistingFunction"));
  
  // Test function calling
  auto add_func = Registry::Global()->Get("TestAdd");
  auto multiply_func = Registry::Global()->Get("TestMultiply");
  
  std::vector<PackedValue> args = {PackedValue(5), PackedValue(3)};
  
  PackedValue add_result = add_func(args);
  PackedValue multiply_result = multiply_func(args);
  
  assert(add_result.AsInt() == 8);  // 5 + 3 = 8
  assert(multiply_result.AsInt() == 15);  // 5 * 3 = 15
  
  // Test listing functions
  auto func_names = Registry::Global()->ListNames();
  assert(func_names.size() >= 2);  // At least our two test functions
  
  bool found_add = false;
  bool found_multiply = false;
  
  for (const auto& name : func_names) {
    if (name == "TestAdd") found_add = true;
    if (name == "TestMultiply") found_multiply = true;
  }
  
  assert(found_add);
  assert(found_multiply);
  
  std::cout << "All C++ registry tests passed!" << std::endl;
  return 0;
}