#pragma once

#include <cuda_fp16.h>

template<typename T>
__device__ __forceinline__
T sigmoid(const T x) {
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

template<typename T>
__device__ __forceinline__
T tanh(const T x) {
    return std::tanh(x);
}

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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)

template<>
__device__ __forceinline__
half sigmoid(const half x) {
    return static_cast<half>(1.0) / (static_cast<half>(1.0) + hexp(-x));
}

template<>
__device__ __forceinline__
half tanh(const half x) {
    return std::tanh(float(x));
}

#endif
