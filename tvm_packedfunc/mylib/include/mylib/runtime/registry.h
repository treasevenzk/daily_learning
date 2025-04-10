#ifndef MYLIB_RUNTIME_REGISTRY_H_
#define MYLIB_RUNTIME_REGISTRY_H_

#include <string>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include "mylib/runtime/packed_func.h"

namespace mylib {
namespace runtime {

/*!
 * \brief 全局函数注册表
 */
class Registry {
 public:
  /*!
   * \brief 获取全局注册表单例
   * \return 全局注册表的引用
   */
  static Registry& Global();
  
  /*!
   * \brief 根据名称注册一个全局PackedFunc
   * \param name 函数名称
   * \param f 函数对象
   */
  void Register(const std::string& name, const PackedFunc& f);
  
  /*!
   * \brief 根据名称查找PackedFunc
   * \param name 函数名称
   * \return 找到的函数，如果未找到则返回nullptr
   */
  const PackedFunc* Find(const std::string& name) const;

 private:
  /*! \brief 构造函数私有化，防止直接创建 */
  Registry() = default;
  
  /*! \brief 存储函数的哈希表 */
  std::unordered_map<std::string, PackedFunc> fmap_;
};

/*!
 * \brief 用于注册全局函数的助手类
 */
class FunctionRegistryEntry {
 public:
  /*!
   * \brief 构造函数，注册名称
   * \param name 函数名称
   */
  explicit FunctionRegistryEntry(const std::string& name) : name_(name) {}
  
  /*!
   * \brief 设置函数体，并注册到全局注册表
   * \param f 函数对象
   * \return 自身的引用，用于链式调用
   */
  FunctionRegistryEntry& set_body(const PackedFunc& f) {
    Registry::Global().Register(name_, f);
    return *this;
  }
  
  /*!
   * \brief 设置类型化的函数体
   * 
   * 这个模板函数可以将普通C++函数包装为PackedFunc
   */
  template<typename FType>
  FunctionRegistryEntry& set_body_typed(FType f);
  
 private:
  /*! \brief 函数名称 */
  std::string name_;
};

// 辅助函数，将不同类型的C++函数转换为PackedFunc

// 两个整数参数的函数 (int, int) -> R
template<typename R>
inline PackedFunc PackFuncTyped(R (*f)(int, int)) {
  return PackedFunc([f](TVMArgs args, TVMRetValue* rv) {
    if (args.num_args() != 2) {
      throw std::runtime_error("Incorrect number of arguments, expected 2");
    }
    *rv = f(args.Get<int>(0), args.Get<int>(1));
  });
}

// 字符串和整数参数的函数 (std::string, int) -> R
template<typename R>
inline PackedFunc PackFuncTyped(R (*f)(const std::string&, int)) {
  return PackedFunc([f](TVMArgs args, TVMRetValue* rv) {
    if (args.num_args() != 2) {
      throw std::runtime_error("Incorrect number of arguments, expected 2");
    }
    *rv = f(args.Get<std::string>(0), args.Get<int>(1));
  });
}

// 向量参数的函数 (std::vector<int>) -> int
inline PackedFunc PackFuncTyped(int (*f)(const std::vector<int>&)) {
  return PackedFunc([f](TVMArgs args, TVMRetValue* rv) {
    std::vector<int> vec;
    for (int i = 0; i < args.num_args(); ++i) {
      vec.push_back(args.Get<int>(i));
    }
    *rv = f(vec);
  });
}

// 全局工厂函数，方便注册
inline FunctionRegistryEntry& RegisterFunction(const std::string& name) {
  static FunctionRegistryEntry reg(name);
  return reg;
}

}  // namespace runtime
}  // namespace mylib

#endif  // MYLIB_RUNTIME_REGISTRY_H_