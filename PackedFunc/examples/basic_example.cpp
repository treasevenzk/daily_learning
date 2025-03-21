#include <iostream>
#include <string>

#include "packed_fun.h"
#include "registry.h"

using namespace packedfun;

// Define some example functions
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

// Register functions using the macro
PACKEDFUN_REGISTER_GLOBAL(AddIntegers);
PACKEDFUN_REGISTER_GLOBAL(Greet);

int main() {
  // Get functions from registry
  auto add_func = Registry::Global()->Get("AddIntegers");
  auto greet_func = Registry::Global()->Get("Greet");
  
  // Call AddIntegers
  std::vector<PackedValue> add_args = {PackedValue(5), PackedValue(7)};
  PackedValue add_result = add_func(add_args);
  std::cout << "5 + 7 = " << add_result.AsInt() << std::endl;
  
  // Call Greet
  std::vector<PackedValue> greet_args = {PackedValue("PackedFun User")};
  PackedValue greet_result = greet_func(greet_args);
  std::cout << greet_result.AsString() << std::endl;
  
  return 0;
}