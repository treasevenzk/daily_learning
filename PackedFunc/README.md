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