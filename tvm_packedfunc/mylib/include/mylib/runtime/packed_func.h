#ifndef MYLIB_RUNTIME_PACKED_FUNC_H_
#define MYLIB_RUNTIME_PACKED_FUNC_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <typeinfo>
#include <stdexcept>

namespace mylib {
namespace runtime {

// 类的前向声明
class TVMPODValue_;
class TVMArgs;
class TVMRetValue;
class PackedFunc;

/*!
 * \brief 安全的类型转换
 * \param src 源类型
 * \return 目标类型
 */
template<typename DstType, typename SrcType>
inline DstType TVMPODValue_Cast(const SrcType& src) {
  static_assert(std::is_pod<SrcType>::value, "Only allow POD type.");
  return reinterpret_cast<const DstType&>(src);
}

/*!
 * \brief 用于参数传递的基础POD值类型
 */
class TVMPODValue_ {
 public:
  // 数据存储联合体
  union {
    int64_t v_int64;
    double v_float64;
    void* v_handle;
    const char* v_str;
  } value_;
  
  // 表示当前存储类型的枚举
  enum TypeCode {
    kNull = 0,
    kInt = 1,
    kFloat = 2,
    kHandle = 3,
    kStr = 4,
  } type_code_;
};

/*!
 * \brief 函数参数类型
 */
class TVMArgs {
 public:
  /*! \brief 构造函数 */
  TVMArgs(const TVMPODValue_* values, const int* type_codes, int num_args)
      : values_(values), type_codes_(type_codes), num_args_(num_args) {}
  
  /*! \brief 获取参数个数 */
  int num_args() const { return num_args_; }
  
  /*! \brief 获取第i个参数的值 */
  template<typename T>
  T Get(int i) const;
  
 private:
  const TVMPODValue_* values_;
  const int* type_codes_;
  int num_args_;
};

/*!
 * \brief PackedFunc是一种可以封装任何函数签名的类型擦除的函数
 *
 * 可以通过TVMArgs和TVMRetValue传递和返回多种类型的参数
 */
class PackedFunc {
 public:
  /*!
   * \brief 函数签名：从TVMArgs接收参数并返回TVMRetValue
   */
  using FType = std::function<void(TVMArgs args, TVMRetValue* rv)>;
  
  /*! \brief 默认构造函数 */
  PackedFunc() = default;
  
  /*! \brief 通过函数构造 */
  explicit PackedFunc(FType f) : body_(f) {}
  
  /*!
   * \brief 调用PackedFunc
   * \param args 函数参数
   * \param rv 函数返回值
   */
  void CallPacked(TVMArgs args, TVMRetValue* rv) const {
    if (body_) body_(args, rv);
  }
  
  /*! \brief 操作符重载，可以像普通函数一样调用 */
  template<typename... Args>
  TVMRetValue operator()(Args&&... args) const;
  
 private:
  /*! \brief 函数主体 */
  FType body_;
};

/*!
 * \brief 函数返回值类型
 */
class TVMRetValue : public TVMPODValue_ {
 public:
  /*! \brief 默认构造函数 */
  TVMRetValue() { type_code_ = kNull; }
  
  /*! \brief 设置返回值为整数 */
  void operator=(int64_t value) {
    type_code_ = kInt;
    value_.v_int64 = value;
  }
  
  // 添加int类型的重载，避免类型转换歧义
  void operator=(int value) {
    type_code_ = kInt;
    value_.v_int64 = static_cast<int64_t>(value);
  }
  
  /*! \brief 设置返回值为浮点数 */
  void operator=(double value) {
    type_code_ = kFloat;
    value_.v_float64 = value;
  }
  
  /*! \brief 设置返回值为字符串 */
  void operator=(const std::string& value) {
    type_code_ = kStr;
    // 在实际实现中需要做内存管理
    value_.v_str = value.c_str();
  }
  
  /*! \brief 设置返回值为函数 */
  void operator=(const PackedFunc& f) {
    // 在实际实现中需要做适当的类型管理
    type_code_ = kHandle;
    value_.v_handle = const_cast<PackedFunc*>(&f);
  }
  
  /*! \brief 获取整数值 */
  int64_t AsInt64() const {
    if (type_code_ != kInt) throw std::invalid_argument("Expected int");
    return value_.v_int64;
  }
  
  /*! \brief 获取浮点数值 */
  double AsDouble() const {
    if (type_code_ != kFloat) throw std::invalid_argument("Expected float");
    return value_.v_float64;
  }
  
  /*! \brief 获取字符串值 */
  std::string AsString() const {
    if (type_code_ != kStr) throw std::invalid_argument("Expected string");
    return std::string(value_.v_str);
  }
  
  /*! \brief 获取函数值 */
  PackedFunc AsFunction() const {
    if (type_code_ != kHandle) throw std::invalid_argument("Expected function");
    return *static_cast<PackedFunc*>(value_.v_handle);
  }
};

}  // namespace runtime
}  // namespace mylib

#endif  // MYLIB_RUNTIME_PACKED_FUNC_H_