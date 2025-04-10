# TVM PackedFunc机制学习项目

编译流程
git clone xxx
mkdir build
cd build
cmake ..
make -j20
将库文件复制到Python包目录
cd ..
cd python
mkdir packedfun/_lib
cp ../build/lib/libpackedfun*.so packedfun/_lib/
运行测试
cd ../tests/python
python test_function_call.py


运行test_function_call.py测试文件，整个项目的流程
### 第一部分：项目结构和初始化流程
当运行```test_function_call.py```时，整个系统的工作流程如下：
1. Python初始化阶段
首先，Python解释器加载```test_function_call.py```,其中导入了```packedfun```模块:
```
import packedfun
from packedfun._ffi import get_global_func, list_global_func
```
这个导入触发一系列模块加载：
1.首先加载```packedfun/__init__.py```,它包含
```
from . import _ffi
```
2.然后加载```packedfun/_ffi/__init__.py```,它导入几个关键组件
```
from .base import PackedFunError
from .function import PackedFunction
from .registry import get_global_func, register_func, list_global_func
```
3.这导致加载```base.py```,此文件包含了核心的C++库加载逻辑:
```
def _load_lib():
    # 查找并加载c++库文件
    # 返回ctypes.CDLL对象
_LIB = _load_lib()
```
```_load_lib```函数会在多个位置查找```libpackedfun_c_api.so```库文件，当找到时，使用```ctypes.CDLL()```加载它。这实际上是在动态加载我们编译的C++代码
2. C++库加载阶段
当```ctypes.CDLL()```加载```libpackedfun_c_api.so```时，操作系统的动态链接会
1.首先加载```libpackedfun_c_api.so```
2.发现它依赖于```libpackedfun.so```，所以也会加载该库
3.解析所有外部符号引用
加载的C++库包含:
* ```src/packed_fun.cpp```-定义PackedValue类
* ```src/registry.cpp```-定义Registry类和相关函数
* ```src/c_api/packed_fun_api.cpp```-定义供Python调用的C API函数
* ```examples/test_functions/test_functions.cpp```-定义我们的测试函数(AddIntegers，Greet)
加载库时，会执行静态初始化部分，包括：
```
PACKEDFUN_REGISTER_GLOBAL(AddIntegers);
PACKEDFUN_REGISTER_GLOBAL(Greet);
```
这些宏展开为静态初始化代码，会调用```Registry::Global()->Register()```将函数注册到全局注册表中
3. Python FFI初始化
加载完C++库后，Python会:
1.定义C API函数的原型(在base.py中)
```
_LIB.PackedFunCreateInt.argtypes = [ctypes.c_int]
_LIB.PackedFunCreateInt.restype = ctypes.POINTER(PackedValueHandle)
```
2.初始化function.py中的```PackedFunction```类，它封装了对C++函数调用
3.初始化registry.py中的函数，提供对C++ Registry的访问
### 测试执行流程
当```test_function_call.py```执行测试时：
1. 测试函数列表获取
```
def test_list_functions(self):
    func_names = list_global_func()
    self.assertIsInstance(func_names, list)
```
这段代码调用了```list_global_func()```,它的执行流程是:<br>
1.Python:```registry.py```中的```list_global_func()```<br>
2.C++: 调用```PackedFunListGlobalFuncNames()```(在```packed_fun_api.cpp```中)<br>
3.C++：该函数使用```Registry::Global()->ListNames()```获取所有注册的函数名<br>
4.C++→Python: 返回字符串数组给Python<br>

2. 测试函数获取
```
def test_add_integers(self):
    add_func = get_global_func("AddIntegers")
    self.assertIsNotNone(add_func)
```
执行流程:<br>
1.Python:registry.py中的get_global_func("AddIntegers")<br>
2.C++:调用```PackedFunGetGlobalFunc("AddIntegers")(在packed_fun_api.cpp中)<br>
3.C++:该函数使用```Registry::Global()->Get("AddIntegers")```获取函数<br>
4.C++→Python:返回函数句柄给Python<br>
5.Python：使用这个句柄创建一个```PackedFunction```对象<br>

3. 测试函数调用
```
def test_add_integers(self):
    add_func = get_global_func("AddIntegers")
    self.assertIsNotNone(add_func)
    result = add_func(10, 20)
    self.assertEqual(result, 30)
```
调用```add_func(10, 20)```的执行流程:<br>
1.Python:```function.py```中的```PackedFunction.__call__(10, 20)```<br>
2.Python→C++:<br>
    * 将Python的```10```转换为C++的```PackedValue```(通过```PackedFunCreateInt```)<br>
    * 将Python的```20```转换为C++的```PackedValue```<br>
    * 准备参数组<br>
3.C++：调用```PackedFunCallFunc```(在```packed_fun_api.cpp```中)<br>
4.C++：该函数找到```AddIntegers```函数并调用它<br>
5.C++：```AddIntegers```函数(在```test_functions.cpp```中)执行```5+7```计算<br>
6.C++→Python：将结果转换回Python值返回<br>