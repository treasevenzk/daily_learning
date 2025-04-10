#include "registry.h"
#include <stdexcept>

namespace packedfun {

Registry* Registry::Global() {
  static Registry instance;
  return &instance;
}

void Registry::Register(const std::string& name, const PackedFunc& func) {
  if (registry_.find(name) != registry_.end()) {
    throw std::runtime_error("Function '" + name + "' already registered");
  }
  registry_[name] = func;
}

PackedFunc Registry::Get(const std::string& name) const {
  auto it = registry_.find(name);
  if (it == registry_.end()) {
    throw std::runtime_error("Function '" + name + "' not found in registry");
  }
  return it->second;
}

bool Registry::Contains(const std::string& name) const {
  return registry_.find(name) != registry_.end();
}

std::vector<std::string> Registry::ListNames() const {
  std::vector<std::string> names;
  names.reserve(registry_.size());
  for (const auto& entry : registry_) {
    names.push_back(entry.first);
  }
  return names;
}

}  // namespace packedfun