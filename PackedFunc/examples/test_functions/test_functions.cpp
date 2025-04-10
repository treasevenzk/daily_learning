#include "packed_fun.h"
#include "registry.h"
#include <string>
#include <vector>
#include <stdexcept>  // 添加这个头文件

using namespace packedfun;

// Define test functions
PackedValue AddIntegers(const std::vector<PackedValue>& args) {
  if (args.size() != 2) {
    throw std::runtime_error("AddIntegers requires exactly 2 arguments");
  }
  
  int a = args[0].AsInt();
  int b = args[1].AsInt();
  
  return PackedValue(a + b);
}

PackedValue Greet(const std::vector<PackedValue>& args) {
  if (args.size() != 1) {
    throw std::runtime_error("Greet requires exactly 1 argument");
  }
  
  std::string name = args[0].AsString();
  std::string greeting = "Hello, " + name + "!";
  
  return PackedValue(greeting);
}

// Register functions
PACKEDFUN_REGISTER_GLOBAL(AddIntegers);
PACKEDFUN_REGISTER_GLOBAL(Greet);