// packed_func.h
#pragma once
#include <functional>
#include <map>
#include <string>
#include <any>
#include <vector>

// 参数包装类
class PackedArgs {
public:
    std::vector<std::any> args;
    
    template<typename T>
    T Get(size_t index) const {
        return std::any_cast<T>(args[index]);
    }
};

// 返回值包装类
class RetValue {
    std::any value;
public:
    // 修改 operator= 的实现
    template<typename T>
    RetValue& operator=(const T& v) {
        value = v;
        return *this;
    }
    
    template<typename T>
    T As() const {
        return std::any_cast<T>(value);
    }
};

// PackedFunc定义
using PackedFunc = std::function<void(PackedArgs args, RetValue* rv)>;

// 全局注册表
class Registry {
    static std::map<std::string, PackedFunc>& Global() {
        static std::map<std::string, PackedFunc> registry;
        return registry;
    }
public:
    static void Register(const std::string& name, PackedFunc func) {
        Global()[name] = func;
    }
    
    static PackedFunc Get(const std::string& name) {
        auto it = Global().find(name);
        if (it != Global().end()) {
            return it->second;
        }
        return nullptr;
    }
};