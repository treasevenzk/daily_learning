#ifndef PACKEDFUN_PACKED_FUN_H_
#define PACKEDFUN_PACKED_FUN_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace packedfun {

// TypeCode for supported data types
enum class TypeCode : int {
  kInt = 0,
  kFloat = 1,
  kStr = 2,
  kBytes = 3,
  kHandle = 4,
  kNull = 5,
  kNodeHandle = 6
};

// PackedValue to represent any value
class PackedValue {
 public:
  // Constructors for different types
  PackedValue() : type_code_(TypeCode::kNull), value_ptr_(nullptr) {}
  
  explicit PackedValue(int value);
  explicit PackedValue(float value);
  explicit PackedValue(const std::string& value);
  
  // Copy and move constructors
  PackedValue(const PackedValue& other);
  PackedValue(PackedValue&& other) noexcept;
  
  // Assignment operators
  PackedValue& operator=(const PackedValue& other);
  PackedValue& operator=(PackedValue&& other) noexcept;

  // Destructor
  ~PackedValue();
  
  // Convert to C++ types
  int AsInt() const;
  float AsFloat() const;
  std::string AsString() const;
  
  // Get type information
  TypeCode type_code() const { return type_code_; }
  
 private:
  TypeCode type_code_;
  void* value_ptr_;
};

// PackedFunc type
using PackedFunc = std::function<PackedValue(const std::vector<PackedValue>&)>;

}  // namespace packedfun

#endif  // PACKEDFUN_PACKED_FUN_H_