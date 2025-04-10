#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义类型码（对应 Python 的 ArgTypeCode）
typedef enum {
    TYPE_INT = 0,
    TYPE_FLOAT = 1,
    TYPE_STRING = 2,
    TYPE_HANDLE = 3
} ArgTypeCode;

// 联合体，存储不同类型的值
typedef union {
    long long v_int64;
    double v_float64;
    char* v_str;
    void* v_handle;
} Value;

// 包含类型码和值的结构体
typedef struct {
    ArgTypeCode type_code;
    Value value;
} ArgValue;

// 模拟一个 C 函数，返回不同类型的值
ArgValue get_value(int choice) {
    ArgValue result;
    switch (choice) {
        case 0:
            result.type_code = TYPE_INT;
            result.value.v_int64 = 42;
            break;
        case 1:
            result.type_code = TYPE_FLOAT;
            result.value.v_float64 = 3.14159;
            break;
        case 2:
            result.type_code = TYPE_STRING;
            result.value.v_str = strdup("Hello from C!");
            break;
        case 3:
            result.type_code = TYPE_HANDLE;
            int* data = malloc(sizeof(int));
            *data = 123;
            result.value.v_handle = data;
            break;
        default:
            result.type_code = TYPE_INT;
            result.value.v_int64 = -1;
    }
    return result;
}

// 释放字符串和句柄内存的函数
void free_value(ArgValue arg) {
    if (arg.type_code == TYPE_STRING && arg.value.v_str != NULL) {
        free(arg.value.v_str);
    } else if (arg.type_code == TYPE_HANDLE && arg.value.v_handle != NULL) {
        free(arg.value.v_handle);
    }
}