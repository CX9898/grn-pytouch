#pragma once

#include <cuda_fp16.h>


namespace dev {

template<typename T>
__device__ __forceinline__
T sigmoid(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + __expf(-x));
    } else if constexpr (std::is_same_v<T, double>) {
        return 1.0 / (1.0 + exp(-x));
    } else if constexpr (std::is_same_v<T, int8_t>) {
        float xf = static_cast<float>(x);
        float yf = 1.0f / (1.0f + __expf(-xf));
        int32_t scaled = static_cast<int32_t>(yf * 127);
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        return static_cast<int8_t>(scaled);
    } else if constexpr (std::is_same_v<T, __half>) {
        float xf = __half2float(x);
        float yf = 1.0f / (1.0f + __expf(-xf));
        return __float2half(yf);
    }
}

template<typename T>
__device__ __forceinline__
T tanh(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return tanhf(x);
    } else if constexpr (std::is_same_v<T, double>) {
        return tanh(x);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        float xf = static_cast<float>(x);
        float yf = tanhf(xf);
        int32_t scaled = static_cast<int32_t>(yf * 127);
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        return static_cast<int8_t>(scaled);
    } else if constexpr (std::is_same_v<T, __half>) {
        float xf = __half2float(x);
        float yf = tanhf(xf);
        return __float2half(yf);
    }
}

} // dev namespace


template<typename T>
__device__ __forceinline__
T d_sigmoid(const T sigmoid_output) {
    return sigmoid_output * (static_cast<T>(1.0) - sigmoid_output);
}

template<typename T>
__device__ __forceinline__
T d_tanh(const T tanh_output) {
    return (static_cast<T>(1.0) - tanh_output * tanh_output);
}

template<typename T>
__device__ __forceinline__
void atomicAddCustom(T* address, T val);

// float specialization
template<>
__device__ __forceinline__
void atomicAddCustom(float* address, float val) {
    atomicAdd(address, val);
}

// double specialization
template<>
__device__ __forceinline__
void atomicAddCustom(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    // 用 CAS 实现低架构 double 原子加法
#endif
}

// int32_t specialization
template<>
__device__ __forceinline__
void atomicAddCustom(int32_t* address, int32_t val) {
    atomicAdd(address, val);
}

// int8_t specialization
template<>
__device__ __forceinline__
void atomicAddCustom(int8_t* address, int8_t val) {
    int32_t* base = reinterpret_cast<int32_t*>(address);
    atomicAdd(base, static_cast<int32_t>(val));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

__device__ __forceinline__
double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
//
//template<>
//__device__ __forceinline__
//half sigmoid(const half x) {
//    return static_cast<half>(1.0) / (static_cast<half>(1.0) + hexp(-x));
//}
//
//template<>
//__device__ __forceinline__
//half tanh(const half x) {
//    return std::tanh(float(x));
//}
//
//#endif
