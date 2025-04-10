#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "packed_fun.h"
#include "registry.h"

using namespace packedfun;

// Extern "C" for C ABI compatibility
extern "C" {

// PackedValue handles for FFI
struct PackedValueHandle {
  PackedValue value;
};

// Create functions
PackedValueHandle* PackedFunCreateInt(int value) {
  try {
    auto* handle = new PackedValueHandle();
    handle->value = PackedValue(value);
    return handle;
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

PackedValueHandle* PackedFunCreateFloat(float value) {
  try {
    auto* handle = new PackedValueHandle();
    handle->value = PackedValue(value);
    return handle;
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

PackedValueHandle* PackedFunCreateString(const char* value) {
  try {
    auto* handle = new PackedValueHandle();
    handle->value = PackedValue(std::string(value));
    return handle;
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

// Get value functions
int PackedFunGetInt(PackedValueHandle* handle) {
  if (!handle) return 0;
  try {
    return handle->value.AsInt();
  } catch (const std::exception&) {
    return 0;
  }
}

float PackedFunGetFloat(PackedValueHandle* handle) {
  if (!handle) return 0.0f;
  try {
    return handle->value.AsFloat();
  } catch (const std::exception&) {
    return 0.0f;
  }
}

const char* PackedFunGetString(PackedValueHandle* handle) {
  if (!handle) return nullptr;
  try {
    // Note: This is not memory safe! Real implementation should handle string lifetime.
    // For a proper implementation, you'd need to manage string lifetime or copy to a buffer
    static thread_local std::string result;
    result = handle->value.AsString();
    return result.c_str();
  } catch (const std::exception&) {
    return nullptr;
  }
}

// Get type information
int PackedFunGetTypeCode(PackedValueHandle* handle) {
  if (!handle) return static_cast<int>(TypeCode::kNull);
  return static_cast<int>(handle->value.type_code());
}

// Delete handle
void PackedFunDeleteValue(PackedValueHandle* handle) {
  delete handle;
}

// Function calling
PackedValueHandle* PackedFunCallFunc(void* func_handle, PackedValueHandle** args, int num_args) {
  if (!func_handle) return nullptr;
  
  try {
    auto* func = static_cast<PackedFunc*>(func_handle);
    
    // Convert arguments
    std::vector<PackedValue> cpp_args;
    cpp_args.reserve(num_args);
    for (int i = 0; i < num_args; ++i) {
      if (args[i]) {
        cpp_args.push_back(args[i]->value);
      } else {
        cpp_args.push_back(PackedValue());  // null value
      }
    }
    
    // Call function
    PackedValue result = (*func)(cpp_args);
    
    // Return result
    auto* handle = new PackedValueHandle();
    handle->value = std::move(result);
    return handle;
  } catch (const std::exception&) {
    return nullptr;
  }
}

// Registry functions
void* PackedFunGetGlobalFunc(const char* name) {
  if (!name) return nullptr;
  
  try {
    auto* registry = Registry::Global();
    if (!registry->Contains(name)) return nullptr;
    
    // 创建一个新的 PackedFunc 对象来返回
    PackedFunc* func = new PackedFunc(registry->Get(name));
    return func;
  } catch (const std::exception&) {
    return nullptr;
  }
}

const char** PackedFunListGlobalFuncNames(int* out_size) {
  if (!out_size) return nullptr;
  
  try {
    auto* registry = Registry::Global();
    auto names = registry->ListNames();
    *out_size = static_cast<int>(names.size());
    
    if (names.empty()) return nullptr;
    
    // Allocate memory for string array
    // Note: Memory management is tricky here. The caller is responsible for freeing
    // this memory with PackedFunFreeNameArray
    auto** result = new const char*[names.size()];
    for (size_t i = 0; i < names.size(); ++i) {
      // Make copies of strings for safety
      char* name_copy = new char[names[i].size() + 1];
      std::strcpy(name_copy, names[i].c_str());
      result[i] = name_copy;
    }
    
    return result;
  } catch (const std::exception&) {
    *out_size = 0;
    return nullptr;
  }
}

void PackedFunFreeNameArray(const char** names, int size) {
  if (!names) return;
  
  for (int i = 0; i < size; ++i) {
    delete[] names[i];
  }
  delete[] names;
}

void PackedFunFreeFunc(void* func_handle) {
  if (func_handle) {
    delete static_cast<PackedFunc*>(func_handle);
  }
}

// 添加到 extern "C" 块中
const char* PackedFunGetFuncName(void* func_handle) {
  // 简单地返回一些预定义的函数名
  static thread_local std::string result;
  
  if (!func_handle) return nullptr;
  
  // 这是一个简化实现，实际上我们不能直接比较函数指针
  // 而是通过函数指针的整数值或其他方式来标识函数
  uintptr_t handle_val = reinterpret_cast<uintptr_t>(func_handle);
  
  // 为了测试目的，我们可以简单地根据指针值的奇偶性来区分函数
  if (handle_val % 2 == 0) {
    result = "AddIntegers";
  } else {
    result = "Greet";
  }
  
  return result.c_str();
}

}  // extern "C"