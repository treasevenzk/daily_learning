#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <random>
#include <vector>
#include <memory>

// 辅助函数：生成随机数据
void generateRandomData(std::vector<float>& data, std::vector<float>& thresholds, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    for(int i = 0; i < size; i++) {
        data[i] = dis(gen);
        thresholds[i] = dis(gen);
    }
}

// 标量版本：逐个比较
void scalarCompare(const std::vector<float>& features, 
                  const std::vector<float>& thresholds, 
                  std::vector<int>& results) {
    for(size_t i = 0; i < features.size(); i++) {
        results[i] = (features[i] < thresholds[i]) ? 1 : 0;
    }
}

// 向量化版本：使用SSE指令一次比较4个值
void vectorizedCompare(const std::vector<float>& features, 
                      const std::vector<float>& thresholds, 
                      std::vector<int>& results) {
    size_t vectorSize = features.size() / 4 * 4;  // 确保是4的倍数

    for(size_t i = 0; i < vectorSize; i += 4) {
        // 加载4个float数据
        __m128 vfeatures = _mm_loadu_ps(&features[i]);
        __m128 vthresholds = _mm_loadu_ps(&thresholds[i]);
        
        // 执行比较操作
        __m128 vcmp = _mm_cmplt_ps(vfeatures, vthresholds);
        
        // 将结果转换为整数并存储
        int mask = _mm_movemask_ps(vcmp);
        for(int j = 0; j < 4; j++) {
            results[i + j] = (mask & (1 << j)) ? 1 : 0;
        }
    }
    
    // 处理剩余的元素
    for(size_t i = vectorSize; i < features.size(); i++) {
        results[i] = (features[i] < thresholds[i]) ? 1 : 0;
    }
}

// 验证结果是否一致
bool verifyResults(const std::vector<int>& result1, 
                  const std::vector<int>& result2) {
    if (result1.size() != result2.size()) return false;
    for(size_t i = 0; i < result1.size(); i++) {
        if(result1[i] != result2[i]) {
            std::cout << "Mismatch at index " << i << ": "
                      << result1[i] << " vs " << result2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 测试函数
void runTest(int size, int iterations) {
    std::cout << "\nRunning test with size = " << size 
              << ", iterations = " << iterations << std::endl;
    
    // 使用vector替代原始数组
    std::vector<float> features(size);
    std::vector<float> thresholds(size);
    std::vector<int> results_scalar(size);
    std::vector<int> results_vector(size);
    
    // 生成测试数据
    generateRandomData(features, thresholds, size);
    
    // 预热
    scalarCompare(features, thresholds, results_scalar);
    vectorizedCompare(features, thresholds, results_vector);
    
    // 测试标量版本
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        scalarCompare(features, thresholds, results_scalar);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>
                      (end - start).count();
    
    // 测试向量化版本
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        vectorizedCompare(features, thresholds, results_vector);
    }
    end = std::chrono::high_resolution_clock::now();
    auto vector_time = std::chrono::duration_cast<std::chrono::microseconds>
                      (end - start).count();
    
    // 验证结果
    bool results_match = verifyResults(results_scalar, results_vector);
    
    // 输出结果
    std::cout << "Scalar version time: " << scalar_time << " microseconds" << std::endl;
    std::cout << "Vector version time: " << vector_time << " microseconds" << std::endl;
    if (scalar_time > 0) {  // 避免除以0
        std::cout << "Speedup: " << static_cast<float>(scalar_time) / vector_time 
                  << "x" << std::endl;
    }
    std::cout << "Results " << (results_match ? "match" : "don't match") << std::endl;
    
    // 如果结果不匹配，输出一些样本数据
    if (!results_match && size > 0) {
        std::cout << "\nSample data for first few elements:" << std::endl;
        for(int i = 0; i < std::min(5, size); i++) {
            std::cout << "Index " << i << ": " 
                     << "Feature=" << features[i] 
                     << " Threshold=" << thresholds[i]
                     << " Scalar=" << results_scalar[i]
                     << " Vector=" << results_vector[i] << std::endl;
        }
    }
}

int main() {
    std::vector<int> test_sizes = {1000, 10000, 100000, 1000000};
    int iterations = 1000;
    
    for(int size : test_sizes) {
        runTest(size, iterations);
    }
    
    return 0;
}