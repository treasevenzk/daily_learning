#ifndef PACKEDFUN_REGISTRY_H_
#define PACKEDFUN_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "packed_fun.h"

namespace packedfun {

// Function registry to store and retrieve functions
class Registry {
 public:
  // Get singleton instance
  static Registry* Global();

  // Register a new function
  void Register(const std::string& name, const PackedFunc& func);
  
  // Get a registered function
  PackedFunc Get(const std::string& name) const;
  
  // Check if a function exists
  bool Contains(const std::string& name) const;
  
  // List all registered functions
  std::vector<std::string> ListNames() const;

 private:
  Registry() = default;
  
  // Map to store functions
  std::unordered_map<std::string, PackedFunc> registry_;
};

// Helper macro to register functions
#define PACKEDFUN_REGISTER_GLOBAL(FunctionName)                              \
  static auto __make_## FunctionName ## _registerer__ = []() {               \
    Registry::Global()->Register(#FunctionName, FunctionName);               \
    return true;                                                            \
  }();

}  // namespace packedfun

#endif  // PACKEDFUN_REGISTRY_H_