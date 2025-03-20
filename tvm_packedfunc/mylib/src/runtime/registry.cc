#include "mylib/runtime/registry.h"
#include <utility>
#include <string>
#include <stdexcept>

namespace mylib {
namespace runtime {

Registry& Registry::Global() {
  static Registry instance;
  return instance;
}

void Registry::Register(const std::string& name, const PackedFunc& f) {
  fmap_[name] = f;
}

const PackedFunc* Registry::Find(const std::string& name) const {
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return nullptr;
  return &(it->second);
}

// set_body_typed的实现
template<typename FType>
FunctionRegistryEntry& FunctionRegistryEntry::set_body_typed(FType f) {
  return set_body(PackFuncTyped(f));
}

// 显式实例化常用的set_body_typed版本
template FunctionRegistryEntry& 
FunctionRegistryEntry::set_body_typed<int(*)(int, int)>(int(*)(int, int));

template FunctionRegistryEntry& 
FunctionRegistryEntry::set_body_typed<std::string(*)(const std::string&, int)>(
    std::string(*)(const std::string&, int));
    
template FunctionRegistryEntry&
FunctionRegistryEntry::set_body_typed<int(*)(const std::vector<int>&)>(
    int(*)(const std::vector<int>&));

}  // namespace runtime
}  // namespace mylib