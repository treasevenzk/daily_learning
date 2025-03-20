#ifndef MYLIB_API_H_
#define MYLIB_API_H_

#include <string>
#include <vector>

namespace mylib {
int Add(int a, int b);

std::string Repeat(const std::string& input, int n);

int SumVector(const std::vector<int>& vec);
}


#endif  // MYLIB_API_H_