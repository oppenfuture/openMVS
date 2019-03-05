#pragma once
#ifndef _GIPUMA_MANAGED_H_
#define _GIPUMA_MANAGED_H_
// Includes CUDA
#include <cuda_runtime_api.h>
#include "helper_cuda.h"

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) {
      cudaFree(ptr);
  }
};

#endif // _GIPUMA_MANAGED_H_
