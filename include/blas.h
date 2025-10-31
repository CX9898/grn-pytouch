#pragma once

#include <cublas_v2.h>

template<typename T>
struct blas {
  struct set_pointer_mode {
    set_pointer_mode(cublasHandle_t handle) : handle_(handle) {
        cublasGetPointerMode(handle_, &old_mode_);
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }

    ~set_pointer_mode() {
        cublasSetPointerMode(handle_, old_mode_);
    }

   private:
    cublasHandle_t handle_;
    cublasPointerMode_t old_mode_;
  };

  struct enable_tensor_cores {
    enable_tensor_cores(cublasHandle_t handle) : handle_(handle) {
        cublasGetMathMode(handle_, &old_mode_);
        cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    }

    ~enable_tensor_cores() {
        cublasSetMathMode(handle_, old_mode_);
    }

   private:
    cublasHandle_t handle_;
    cublasMath_t old_mode_;
  };
};

inline cublasStatus_t int8_gemm_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const int32_t *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const int32_t *beta,
    void *C, int ldc) {
    return cublasGemmEx(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_8I, lda,
        B, CUDA_R_8I, ldb,
        beta,
        C, CUDA_R_32I, ldc,
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT
    );
}

template<>
struct blas<int8_t> {
  static constexpr auto *gemm = &int8_gemm_wrapper;
};

inline cublasStatus_t int16_gemm_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const int32_t *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const int32_t *beta,
    void *C, int ldc) {
    return cublasGemmEx(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_16I, lda,
        B, CUDA_R_16I, ldb,
        beta,
        C, CUDA_R_32I, ldc,
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT
    );
}

template<>
struct blas<int16_t> {
  static constexpr auto *gemm = &int16_gemm_wrapper;
};

template<>
struct blas<__half> {
  static constexpr decltype(cublasHgemm) *gemm = &cublasHgemm;
};

template<>
struct blas<float> {
  static constexpr decltype(cublasSgemm) *gemm = &cublasSgemm;
};

template<>
struct blas<double> {
  static constexpr decltype(cublasDgemm) *gemm = &cublasDgemm;
};
