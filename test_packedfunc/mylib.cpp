// mylib.cpp
#include "packed_func.h"
#include <vector>
#include <cstring>
#include <cstdarg>

// 确保函数被导出
#ifdef __cplusplus
extern "C" {
#endif

// 存储函数参数的结构体
struct FunctionArgs {
    std::vector<std::any> args;
};

// 导出的函数声明
__attribute__((visibility("default")))
int CallFunction(const char* name, ...);

__attribute__((visibility("default")))
void InitLibrary();

#ifdef __cplusplus
}
#endif

// 注册一个简单的加法函数
void RegisterAdd() {
    PackedFunc add_func = [](PackedArgs args, RetValue* rv) {
        int a = args.Get<int>(0);
        int b = args.Get<int>(1);
        *rv = a + b;
    };
    Registry::Register("add", add_func);
}

// 注册一个处理向量的函数
void RegisterProcessVector() {
    PackedFunc process_func = [](PackedArgs args, RetValue* rv) {
        auto vec = args.Get<std::vector<double>>(0);
        for(auto& v : vec) {
            v *= 2;
        }
        *rv = vec;
    };
    Registry::Register("process_vector", process_func);
}

// 实现CallFunction
int CallFunction(const char* name, ...) {
    va_list args;
    va_start(args, name);
    
    PackedArgs packed_args;
    
    // 根据函数名来处理参数
    if (strcmp(name, "add") == 0) {
        int a = va_arg(args, int);
        int b = va_arg(args, int);
        packed_args.args.push_back(a);
        packed_args.args.push_back(b);
    }
    else if (strcmp(name, "process_vector") == 0) {
        // 处理vector参数
        double* arr = va_arg(args, double*);
        int size = va_arg(args, int);
        std::vector<double> vec(arr, arr + size);
        packed_args.args.push_back(vec);
    }
    
    va_end(args);
    
    RetValue rv;
    auto func = Registry::Get(name);
    if (func) {
        func(packed_args, &rv);
        return rv.As<int>();
    }
    return 0;
}

// 初始化函数
void InitLibrary() {
    RegisterAdd();
    RegisterProcessVector();
}