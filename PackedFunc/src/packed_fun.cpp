#include "packed_fun.h"

#include <cstring>
#include <stdexcept>

namespace packedfun {

// PackedValue implementation
PackedValue::PackedValue(int value) 
    : type_code_(TypeCode::kInt) {
  auto* ptr = new int(value);
  value_ptr_ = ptr;
}

PackedValue::PackedValue(float value) 
    : type_code_(TypeCode::kFloat) {
  auto* ptr = new float(value);
  value_ptr_ = ptr;
}

PackedValue::PackedValue(const std::string& value) 
    : type_code_(TypeCode::kStr) {
  auto* ptr = new std::string(value);
  value_ptr_ = ptr;
}

PackedValue::PackedValue(const PackedValue& other) 
    : type_code_(other.type_code_), value_ptr_(nullptr) {
  switch (type_code_) {
    case TypeCode::kInt:
      value_ptr_ = new int(*static_cast<int*>(other.value_ptr_));
      break;
    case TypeCode::kFloat:
      value_ptr_ = new float(*static_cast<float*>(other.value_ptr_));
      break;
    case TypeCode::kStr:
      value_ptr_ = new std::string(*static_cast<std::string*>(other.value_ptr_));
      break;
    case TypeCode::kNull:
      value_ptr_ = nullptr;
      break;
    default:
      throw std::runtime_error("Unsupported type code in copy constructor");
  }
}

PackedValue::PackedValue(PackedValue&& other) noexcept
    : type_code_(other.type_code_), value_ptr_(other.value_ptr_) {
  other.type_code_ = TypeCode::kNull;
  other.value_ptr_ = nullptr;
}

PackedValue& PackedValue::operator=(const PackedValue& other) {
  if (this != &other) {
    // Clean up old value
    this->~PackedValue();
    
    // Copy new value
    type_code_ = other.type_code_;
    switch (type_code_) {
      case TypeCode::kInt:
        value_ptr_ = new int(*static_cast<int*>(other.value_ptr_));
        break;
      case TypeCode::kFloat:
        value_ptr_ = new float(*static_cast<float*>(other.value_ptr_));
        break;
      case TypeCode::kStr:
        value_ptr_ = new std::string(*static_cast<std::string*>(other.value_ptr_));
        break;
      case TypeCode::kNull:
        value_ptr_ = nullptr;
        break;
      default:
        throw std::runtime_error("Unsupported type code in assignment operator");
    }
  }
  return *this;
}

PackedValue& PackedValue::operator=(PackedValue&& other) noexcept {
  if (this != &other) {
    // Clean up old value
    this->~PackedValue();
    
    // Move new value
    type_code_ = other.type_code_;
    value_ptr_ = other.value_ptr_;
    
    // Reset other
    other.type_code_ = TypeCode::kNull;
    other.value_ptr_ = nullptr;
  }
  return *this;
}

PackedValue::~PackedValue() {
  switch (type_code_) {
    case TypeCode::kInt:
      delete static_cast<int*>(value_ptr_);
      break;
    case TypeCode::kFloat:
      delete static_cast<float*>(value_ptr_);
      break;
    case TypeCode::kStr:
      delete static_cast<std::string*>(value_ptr_);
      break;
    default:
      // Nothing to clean up for kNull or other types
      break;
  }
  value_ptr_ = nullptr;
}

int PackedValue::AsInt() const {
  if (type_code_ != TypeCode::kInt) {
    throw std::runtime_error("PackedValue is not an integer");
  }
  return *static_cast<int*>(value_ptr_);
}

float PackedValue::AsFloat() const {
  if (type_code_ != TypeCode::kFloat) {
    throw std::runtime_error("PackedValue is not a float");
  }
  return *static_cast<float*>(value_ptr_);
}

std::string PackedValue::AsString() const {
  if (type_code_ != TypeCode::kStr) {
    throw std::runtime_error("PackedValue is not a string");
  }
  return *static_cast<std::string*>(value_ptr_);
}

}  // namespace packedfun