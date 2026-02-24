/**
 * @file cusparse_solver.cu
 * @brief Nangila SPICE cuSPARSE GPU Factorization Backend
 *
 * This file implements the C++ FFI bridge for the Rust `nangila-node` solver.
 * It manages the lifecycle of the NVIDIA cuSPARSE execution context, allocating
 * GPU memory for the Modified Nodal Analysis (MNA) Compressed Sparse Row (CSR)
 * matrix elements across the PCIe bus, and executing extremely fast Sparse
 * Matrix-Vector multiplication (`cusparseSpMV`) for the transient simulation
 * loop.
 */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

extern "C" {
struct GpuSolverContext {
  cusparseHandle_t handle;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  int n;
  int nnz;
  int *d_csrRowPtr;
  int *d_csrColInd;
  double *d_csrVal;
  double *d_x;
  double *d_y;
  void *dBuffer;
};

GpuSolverContext *cusparse_init_solver() {
  GpuSolverContext *ctx = new GpuSolverContext();
  cusparseStatus_t status = cusparseCreate(&ctx->handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "CUSPARSE Library initialization failed" << std::endl;
  }
  ctx->matA = nullptr;
  ctx->vecX = nullptr;
  ctx->vecY = nullptr;
  ctx->n = 0;
  ctx->nnz = 0;
  ctx->d_csrRowPtr = nullptr;
  ctx->d_csrColInd = nullptr;
  ctx->d_csrVal = nullptr;
  ctx->d_x = nullptr;
  ctx->d_y = nullptr;
  ctx->dBuffer = nullptr;
  return ctx;
}

void cusparse_load_matrix(GpuSolverContext *ctx, int n, int nnz,
                          const int *h_csrRowPtr, const int *h_csrColInd,
                          const double *h_csrVal) {
  ctx->n = n;
  ctx->nnz = nnz;

  cudaMalloc((void **)&ctx->d_csrRowPtr, sizeof(int) * (n + 1));
  cudaMalloc((void **)&ctx->d_csrColInd, sizeof(int) * nnz);
  cudaMalloc((void **)&ctx->d_csrVal, sizeof(double) * nnz);
  cudaMalloc((void **)&ctx->d_x, sizeof(double) * n);
  cudaMalloc((void **)&ctx->d_y, sizeof(double) * n);

  cudaMemcpy(ctx->d_csrRowPtr, h_csrRowPtr, sizeof(int) * (n + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_csrColInd, h_csrColInd, sizeof(int) * nnz,
             cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_csrVal, h_csrVal, sizeof(double) * nnz,
             cudaMemcpyHostToDevice);

  cusparseCreateCsr(&ctx->matA, n, n, nnz, ctx->d_csrRowPtr, ctx->d_csrColInd,
                    ctx->d_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  cusparseCreateDnVec(&ctx->vecX, n, ctx->d_x, CUDA_R_64F);
  cusparseCreateDnVec(&ctx->vecY, n, ctx->d_y, CUDA_R_64F);

  double alpha = 1.0;
  double beta = 0.0;
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          ctx->matA, ctx->vecX, &beta, ctx->vecY, CUDA_R_64F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&ctx->dBuffer, bufferSize);
}

void cusparse_spmv(GpuSolverContext *ctx, const double *h_x, double *h_y) {
  cudaMemcpy(ctx->d_x, h_x, sizeof(double) * ctx->n, cudaMemcpyHostToDevice);

  double alpha = 1.0;
  double beta = 0.0;
  cusparseSpMV(ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, ctx->matA,
               ctx->vecX, &beta, ctx->vecY, CUDA_R_64F,
               CUSPARSE_SPMV_ALG_DEFAULT, ctx->dBuffer);

  cudaMemcpy(h_y, ctx->d_y, sizeof(double) * ctx->n, cudaMemcpyDeviceToHost);
}

void cusparse_free_solver(GpuSolverContext *ctx) {
  if (ctx) {
    if (ctx->matA)
      cusparseDestroySpMat(ctx->matA);
    if (ctx->vecX)
      cusparseDestroyDnVec(ctx->vecX);
    if (ctx->vecY)
      cusparseDestroyDnVec(ctx->vecY);
    if (ctx->dBuffer)
      cudaFree(ctx->dBuffer);
    if (ctx->d_csrRowPtr)
      cudaFree(ctx->d_csrRowPtr);
    if (ctx->d_csrColInd)
      cudaFree(ctx->d_csrColInd);
    if (ctx->d_csrVal)
      cudaFree(ctx->d_csrVal);
    if (ctx->d_x)
      cudaFree(ctx->d_x);
    if (ctx->d_y)
      cudaFree(ctx->d_y);
    if (ctx->handle)
      cusparseDestroy(ctx->handle);
    delete ctx;
  }
}
}
