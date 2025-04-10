#include <Python.h>
#include "mylib/runtime/registry.h"

// Python模块初始化函数 - 必须用extern "C"导出
extern "C" PyObject* PyInit_libmylib(void) {
    static PyMethodDef methods[] = {
        {"get_global_func", [](PyObject*, PyObject* args) -> PyObject* {
            const char* name;
            if (!PyArg_ParseTuple(args, "s", &name)) {
                return NULL;
            }
            // 简单实现
            Py_RETURN_NONE;
        }, METH_VARARGS, "Get global function by name"},
        {NULL, NULL, 0, NULL}  // 结束标记
    };

    static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "libmylib",           // 模块名
        "MyLib Python Module", // 模块文档
        -1,                   // 模块状态大小
        methods               // 方法表
    };

    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    return m;
}