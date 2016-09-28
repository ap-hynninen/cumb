/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

//
// Micro-benchmark for memory operations
//

#include <cstdio>
#include "CudaUtils.h"
#include "cumbUtils.h"

__global__ void clearCacheKernel(int* buffer, const int bufferSize) {
  for (int t=threadIdx.x + blockIdx.x*blockDim.x;t < bufferSize;t+=blockDim.x*gridDim.x) {
    buffer[t] = t;
  }
}

template<typename T>
__global__ void memoryTransactionKernel(T* buffer) {
  __shared__ T sh_a;
  int i = threadIdx.x;
  long long int start = clock64();  
  T a = buffer[i];
  sh_a = a;
  long long int end = clock64();
  printf("%d %f\n", (int)(end - start), (float)sh_a);
}

template <typename T, int niter>
__global__ void pChaseKernel(T* array, const int skip, const int offset) {
  
  __shared__ int duration[niter];
  __shared__ T dummy[niter];

  {
    T j = threadIdx.x*skip + offset;
    for (int it=0;it < niter;it++) {
      int start = clock();
      j = array[j];
      dummy[it] = j;
      int end = clock();
      duration[it] = end - start;
    }
  }

  if (threadIdx.x == 0) {
#if 1
    int total_duration = 0;
    int total_duration2 = 0;
    int total_dummy = 0;
    for (int it=1;it < niter;it++) {
      int d = duration[it];
      total_duration += d;
      total_duration2 += d*d;
      total_dummy += (int)dummy[it];
    }
    float avg_duration = (float)total_duration/(float)(niter - 1);
    float avg_duration2 = (float)total_duration2/(float)(niter - 1);
    float std_duration = sqrtf(avg_duration2 - avg_duration*avg_duration);
    printf("%1.2f %1.2f %d\n", avg_duration, std_duration, total_dummy);
#else
    for (int it=0;it < niter;it++) {
      printf("%d %d\n", duration[it], (int)dummy[it]);
    }
#endif
  }
}

template <typename T>
__global__ void pChaseKernel2(T* array, const int skip, const int offset, const int niter) {

  extern __shared__ char dummychar[];
  volatile int* duration = (int*)dummychar;
  volatile T* dummy = (T*)&dummychar[niter*sizeof(int)];
  
  {
    T j = threadIdx.x*skip + offset;
    for (int it=0;it < niter;it++) {
      // int start = clock();
      duration[it] = clock();
      j = array[j];
      dummy[it*blockDim.x + threadIdx.x] = j;
      // int end = clock();
      // duration[it] = end - start;
      duration[it] = clock() - duration[it];
    }
  }

  if (threadIdx.x == 0) {
#if 1
    int total_duration = 0;
    int total_duration2 = 0;
    int total_dummy = 0;
    for (int it=1;it < niter;it++) {
      int d = duration[it];
      total_duration += d;
      total_duration2 += d*d;
      total_dummy += (int)dummy[it];
    }
    float avg_duration = (float)total_duration/(float)(niter - 1);
    float avg_duration2 = (float)total_duration2/(float)(niter - 1);
    float std_duration = sqrtf(avg_duration2 - avg_duration*avg_duration);
    printf("%1.2f %1.2f %d\n", avg_duration, std_duration, total_dummy);
#else
    for (int it=0;it < niter;it++) {
      printf("%d %d\n", duration[it], (int)dummy[it]);
    }
#endif
  }
}

template <typename T>
__global__ void pChaseMaxwellKernel(T* array, const int skip, const int offset, const int niter) {
  
  extern __shared__ char dummychar[];
  T* dummy = (T*)dummychar;

  int start = clock();
  T j = threadIdx.x*skip + offset;
  for (int it=0;it < niter;it++) {
    j = array[j];
    dummy[it] = j;
  }
  int end = clock();
  int duration = (int)(end - start);

  if (threadIdx.x == 0) {
    int max_dummy = 0;
    for (int it=0;it < niter;it++) max_dummy = max(max_dummy, (int)dummy[it]);
    printf("%1.2f %d\n", (float)duration/(float)niter, max_dummy);
  }
}

template <typename T>
__global__ void pChaseMaxwellKernel2(T* array, const int skip, const int offset, const int niter) {
  
  extern __shared__ char dummychar[];
  volatile T* dummy = (T*)dummychar;

  int start = clock();
  T j = threadIdx.x*skip + offset;
  for (int it=0;it < niter;it++) {
    T j0 = j;
    j = array[j];
    // printf("%d %d | %d -> %d\n", it, threadIdx.x, j0, j);
    dummy[it*blockDim.x + threadIdx.x] = j;
  }
  int end = clock();
  int duration = (int)(end - start);

  if (threadIdx.x == 0) {
    int max_dummy = 0;
    for (int it=0;it < niter;it++) {
      for (int t=0;t < blockDim.x;t++) {
        max_dummy = max(max_dummy, (int)dummy[it*blockDim.x + t]);
      }
    }
    printf("%1.2f %d\n", (float)duration/(float)niter, max_dummy);
  }
}

__global__ void writePCLKernel(int* array, const int niter) {
  for (int it=0;it < niter;it++) {
    array[it*32 + threadIdx.x] = 1;
  }
}

__global__ void copyPCLKernel(int* src, int* dst, const int niter) {
  __shared__ int shbuf[32];
  for (int it=0;it < niter;it++) {
    __syncthreads();
    shbuf[threadIdx.x] = src[it*32 + threadIdx.x];
    __syncthreads();
    dst[it*32 + threadIdx.x] = shbuf[threadIdx.x];
  }
}

template <typename T>
__global__ void memoryLatencyKernel(T* bufferIn, T* bufferOut) {
  extern __shared__ int shCycles[];
  // int p = threadIdx.x*(128/sizeof(T)) + 1 + blockIdx.x*1024;
  int p = threadIdx.x;
  //if (threadIdx.x % 32 > 0) p += 128;
  long long int start = clock64();
  T a = bufferIn[p];
  long long int end = clock64();
  shCycles[threadIdx.x] = (int)(end - start);
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int minCycle = (1 << 30);
    int maxCycle = 0;
    int aveCycle = 0;
    for (int i=0;i < blockDim.x;i++) {
      minCycle = min(minCycle, shCycles[i]);
      maxCycle = max(maxCycle, shCycles[i]);
      aveCycle += shCycles[i];
    }
    printf("%d %d %d\n", minCycle, maxCycle, aveCycle/blockDim.x);
  }
  bufferOut[threadIdx.x] = a;
}

template <typename T>
__global__ void memoryLatencyKernel2(T* bufferOut) {
  extern __shared__ int shCycles[];
  int p = threadIdx.x + 1;
  long long int start = clock64();
  bufferOut[p] = 1.2;
  long long int end = clock64();
  shCycles[threadIdx.x] = (int)(end - start);
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int minCycle = (1 << 30);
    int maxCycle = 0;
    int aveCycle = 0;
    for (int i=0;i < blockDim.x;i++) {
      minCycle = min(minCycle, shCycles[i]);
      maxCycle = max(maxCycle, shCycles[i]);
      aveCycle += shCycles[i];
    }
    printf("%d %d %d\n", minCycle, maxCycle, aveCycle/blockDim.x);
  }
}

template <typename T>
__global__ void memoryWriteKernel(T* buffer, const int nwrite, const int stride, const int offset) {
  // for (int t=threadIdx.x + blockIdx.x*blockDim.x;t < nwrite;t+=blockDim.x*gridDim.x) {
    int t = threadIdx.x + blockIdx.x*blockDim.x;
    int wid = t / 32;
    int tid = t % 32;
    // T a = (t*nwrite*blockDim.x) + nwrite*threadIdx.x;
    // buffer[t*stride + offset] = a;
    buffer[wid*stride + tid + offset] = 1;
  // }
}

__global__ void memoryWriteKernel2(char* buffer, const int nwrite, const int stride, const int offset) {
  int t = threadIdx.x + blockIdx.x*blockDim.x;
  int wid = t / 32;
  int tid = t % 32;
  if (tid == 0) buffer[wid*stride + offset] = 1;
}

__global__ void cyclesPerOperationKernel() {
  // int a = threadIdx.x;
  // int b = blockIdx.x;
  long long int start = clock64();
  int a = threadIdx.x;
  int b = blockIdx.x;
  b += a;
  a *= 17;
  b += 3;
  a -= b;
  long long int end = clock64();
  printf("threadIdx.x %d cycles %lld a %d b %d\n", threadIdx.x, end-start, a, b);
}

__global__ void cacheLineKernel(double* buffer, double* res) {
  int t = threadIdx.x;
  //
  double a = buffer[t+1];
  //
  double sum = 0.0;
  for (int i=0;i < 32;i++) {
    sum += __shfl(a, i);
  }
  if (threadIdx.x == 0) res[0] = sum;
}

__global__ void writeVolMmk(const int* pos, const int n, double* buffer) {
  for (int i=threadIdx.x;i < n;i+=blockDim.x) {
    int posval = pos[i];
    buffer[posval] = 1.0;
  }
}

__global__ void readWriteVolMmk(const int* posr, const int* posw, const int n, double* buffer) {
  extern __shared__ double shBuffer[];

  __syncthreads();

  for (int i=threadIdx.x;i < n;i+=blockDim.x) {
    int posr_val = posr[i];
    shBuffer[i] = buffer[posr_val];
  }

  __syncthreads();

  for (int i=threadIdx.x;i < n;i+=blockDim.x) {
    int posw_val = posw[i];
    buffer[posw_val] = shBuffer[i];
  }
}

// ############################################################################
// ############################################################################
// ############################################################################

static int SM_major = 0;

void volMmk(int nthread, char* filebase);
void copyPCL(int nthread, int niter);
void writePCL(int nthread, int niter);
template <typename T> void pChase(int nthread, int stride, int offset);
void memoryTransactions();
template <typename T> void memoryLatency(int nwarp, int nsm);
void clearCache(int* buffer, const int bufferSize);
void cyclesPerOperation();
template <typename T> void memoryWrite(int stride, int offset);
void memoryWrite2(int stride, int offset);
void cacheLine();
void printDeviceInfo();

int main(int argc, char *argv[]) {

  int nthread = 1;
  int stride = 1;
  int offset = 0;
  int niter = 0;
  int deviceID = 0;
  int size = 4;
  char file[64];
  bool arg_ok = true;
  if (argc >= 2) {
    int i = 1;
    while (i < argc) {
      if (strcmp(argv[i], "-stride") == 0) {
        sscanf(argv[i+1], "%d", &stride);
        i += 2;
      } else if (strcmp(argv[i], "-nthread") == 0) {
        sscanf(argv[i+1], "%d", &nthread);
        i += 2;
      } else if (strcmp(argv[i], "-offset") == 0) {
        sscanf(argv[i+1], "%d", &offset);
        i += 2;
      } else if (strcmp(argv[i], "-size") == 0) {
        sscanf(argv[i+1], "%d", &size);
        i += 2;
      } else if (strcmp(argv[i], "-niter") == 0) {
        sscanf(argv[i+1], "%d", &niter);
        i += 2;
      } else if (strcmp(argv[i], "-file") == 0) {
        sscanf(argv[i+1], "%s", file);
        i += 2;
      } else if (strcmp(argv[i], "-device") == 0) {
        sscanf(argv[i+1], "%d", &deviceID);
        i += 2;
      } else {
        arg_ok = false;
        break;
      }
    }
  } else if (argc > 1) {
    arg_ok = false;
  }

  if (!arg_ok) {
    printf("cumb [options]\n");
    printf("Options:\n");
    printf("-nthread [nthread]\n");
    printf("-stride [stride]\n");
    printf("-offset [offset]\n");
    printf("-size [size]\n");
    printf("-niter [niter]\n");
    printf("-file [file]\n");
    printf("-device [device]\n");
    return 1;
  }

  cudaCheck(cudaSetDevice(deviceID));
  printDeviceInfo();

  int* buffer = NULL;
  int bufferSize = 1000000;
  allocate_device<int>(&buffer, bufferSize);

  // for (int i=1;i <= 1;i++) {
  //   clearCache(buffer, bufferSize);
  //   memoryLatency<long long int>(i, 1);
  // }

  // clearCache(buffer, bufferSize);
  // memoryLatency<int>(1);

  // clearCache(buffer, bufferSize);
  // memoryWrite2(stride, offset);

  // clearCache(buffer, bufferSize);
  // memoryTransactions();

  // clearCache(buffer, bufferSize);
  // writePCL(nthread, niter);

  // clearCache(buffer, bufferSize);
  // copyPCL(nthread, niter);

  // volMmk(nthread, file);

#if 1
  if (nthread == 0) {
    for (int i=1;i <= 32;i++) {
      clearCache(buffer, bufferSize);
      if (size == 4)
        pChase<int>(i, stride, offset);
      else
        pChase<long long int>(i, stride, offset);
    }
  } else {
    clearCache(buffer, bufferSize);
    if (size == 4)
      pChase<int>(nthread, stride, offset);
    else
      pChase<long long int>(nthread, stride, offset);
  }
#endif

  // clearCache(buffer, bufferSize);
  // pChase<long long int>(stride);

  // clearCache(buffer, bufferSize);
  // memoryWrite<long long int>(stride, offset);

  // clearCache(buffer, bufferSize);
  // cacheLine();

  deallocate_device<int>(&buffer);

  // cyclesPerOperation();

  cudaCheck(cudaDeviceReset());
  return 0;
}

void volMmk(int nthread, char* filebase) {
  char filename[256];
  sprintf(filename, "%sr.txt", filebase);
  std::vector<int> h_posr = loadPos(filename);
  sprintf(filename, "%sw.txt", filebase);
  std::vector<int> h_posw = loadPos(filename);

  if (h_posr.size() == 0 || h_posr.size() != h_posw.size()) return;

  int* posr;
  int* posw;
  allocate_device<int>(&posr, h_posr.size());
  allocate_device<int>(&posw, h_posw.size());
  copy_HtoD_sync<int>(h_posr.data(), posr, h_posr.size());
  copy_HtoD_sync<int>(h_posw.data(), posw, h_posw.size());

  int maxpos = std::max(h_posr.back(), h_posw.back());
  double* buffer;
  allocate_device<double>(&buffer, maxpos + 1);

  // writeVolMmk<<< 1, nthread >>>(posw, h_posw.size(), buffer);
  readWriteVolMmk<<< 1, nthread, h_posr.size()*sizeof(double) >>>(posr, posw, h_posw.size(), buffer);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  deallocate_device<int>(&posr);
  deallocate_device<int>(&posw);
  deallocate_device<double>(&buffer);
}

void copyPCL(int nthread, int niter) {
  int arraySize = 32*1024*1024;
  int* arraySrc;
  int* arrayDst;
  allocate_device<int>(&arraySrc, arraySize);
  allocate_device<int>(&arrayDst, arraySize);

  copyPCLKernel<<< 1, nthread >>>(arraySrc, arrayDst, niter);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  deallocate_device<int>(&arraySrc);
  deallocate_device<int>(&arrayDst);
}

void writePCL(int nthread, int niter) {
  int arraySize = 32*1024*1024;
  int* array;
  allocate_device<int>(&array, arraySize);

  writePCLKernel<<< 1, nthread >>>(array, niter);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  deallocate_device<int>(&array);
}

//
// Count number of global memory transactions per one request for potentially
// scattered accesses to elements listed in pos
// NOTE: Assumes pos is sorted
//
int glTransactions(const int* pos, const int n, const int accWidth) {
  int count = 0;
  int iseg_prev = -1;
  for (int i=0;i < n;i++) {
    int iseg = pos[i]/accWidth;
    count += (iseg != iseg_prev);
    iseg_prev = iseg;
  }
  return count;
}

//
// Stride = stride in bytes
//
template <typename T>
void pChase(int nthread, int stride, int offset) {
  int niter = 320;
  int skip = stride/sizeof(T);
  int arraySize = (niter + 1)*skip*nthread*100;
  // printf("arraySize %dMB\n", arraySize*sizeof(T)/(1024*1024));
  T* array;
  allocate_device<T>(&array, arraySize);
  T* h_array = new T[arraySize];
  for (int i=0;i < arraySize;i++) {
    h_array[i] = (T)(-1);
  }
  // for (int i=0;i < arraySize;i++) {
  //   int iblock = i/nthread;
  //   int ithread = i % nthread;
  //   h_array[i] = (T)((iblock + stride)*nthread + ithread) % arraySize;
  // }
  for (int i=0;i < niter*skip*nthread;i+=skip) {
    if (skip*nthread + i + offset > arraySize) {
      printf("ERROR\n");
    }
    h_array[i + offset] = (skip*nthread + i + offset) % arraySize;
  }
  
  // int k = 0;
  // for (int j=0;j < 33;j++) {
  //   for (int i=0;i < 32;i++) {
  //     int val = (int)h_array[k++];
  //     if (val < 0) {
  //       printf("X ");
  //     } else {
  //       printf("%d ", val);
  //     }
  //   }
  //   printf("\n");
  // }
  
  // int* pos = new int[nthread];
  // for (int t=0;t < nthread;t++) {
  //   pos[t] = t*skip;
  // }
  // for (int t=0;t < nthread;t++) {
  //   printf("%d ", pos[t]);
  // }
  // printf("\n");

  // int accWidth = 128/sizeof(T);
  // int tran = glTransactions(pos, nthread, accWidth);
  // printf("tran %d\n", tran);

  // delete [] pos;

  copy_HtoD_sync<T>(h_array, array, arraySize);
  cudaCheck(cudaDeviceSynchronize());
  delete [] h_array;

  if (SM_major >= 5) {
    pChaseMaxwellKernel<T> <<< 1, nthread, niter*sizeof(T) >>>(array, skip, offset, niter);
    // pChaseMaxwellKernel2<T> <<< 1, nthread, niter*sizeof(T)*nthread >>>(array, skip, offset, niter);
  } else {
    pChaseKernel<T, 320> <<< 1, nthread >>>(array, skip, offset);
    // pChaseKernel2<T> <<< 1, nthread, niter*sizeof(int) + niter*sizeof(T)*nthread >>>(array, skip, offset, niter);
  }
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  deallocate_device<T>(&array);
}

void cacheLine() {
  double* buffer = NULL;
  allocate_device<double>(&buffer, 256);

  cudaCheck(cudaDeviceSynchronize());

  int nthread = 32;
  int nblock = 1;
  cacheLineKernel <<< nblock, nthread >>>(buffer, &buffer[128]);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  deallocate_device<double>(&buffer);
}

void memoryTransactions() {
  float* buffer = NULL;
  allocate_device<float>(&buffer, 1024);

  int nthread = 1;
  int nblock =1;
  memoryTransactionKernel<float> <<< nblock, nthread >>>(buffer);
  cudaCheck(cudaGetLastError());

  memoryTransactionKernel<float> <<< nblock, nthread >>>(buffer);
  cudaCheck(cudaGetLastError());

  memoryTransactionKernel<float> <<< nblock, nthread >>>(buffer);
  cudaCheck(cudaGetLastError());

  memoryTransactionKernel<double> <<< nblock, nthread >>>((double *)buffer);
  cudaCheck(cudaGetLastError());

  memoryTransactionKernel<double> <<< nblock, nthread >>>((double *)buffer);
  cudaCheck(cudaGetLastError());

  memoryTransactionKernel<double> <<< nblock, nthread >>>((double *)buffer);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());  

  deallocate_device<float>(&buffer);
}

template <typename T>
void memoryWrite(int stride, int offset) {
  int nwrite = 31249408/2;
  int bufferSize = nwrite*34;
  T* buffer = NULL;
  allocate_device<T>(&buffer, bufferSize);
  printf("bufferSize %f GB\n", bufferSize*sizeof(T)/1000000000.0f);

  cudaCheck(cudaDeviceSynchronize());

  int nthread = 512;
  int nblock = nwrite/nthread;
  int numActiveBlock;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, memoryWriteKernel<T>, nthread, 0);
  printf("nthread %d nblock %d numActiveBlock %d\n", nthread, nblock, numActiveBlock);
  memoryWriteKernel<T> <<< nblock, nthread >>>(buffer, nwrite, stride, offset);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  long long int bytesWritten = nwrite*sizeof(T);
  printf("wrote %lld bytes using stride %d and offset %d\n", bytesWritten, stride, offset);

  deallocate_device<T>(&buffer);
}

void memoryWrite2(int stride, int offset) {
  int nwrite = 31249408/2;
  int bufferSize = nwrite*34;
  char* buffer = NULL;
  allocate_device<char>(&buffer, bufferSize);
  printf("bufferSize %f GB\n", bufferSize*sizeof(char)/1000000000.0f);

  cudaCheck(cudaDeviceSynchronize());

  int nthread = 512;
  int nblock = nwrite/nthread;
  int numActiveBlock;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, memoryWriteKernel2, nthread, 0);
  printf("nthread %d nblock %d numActiveBlock %d\n", nthread, nblock, numActiveBlock);
  memoryWriteKernel2 <<< nblock, nthread >>>(buffer, nwrite, stride, offset);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());

  printf("wrote using stride %d and offset %d\n", stride, offset);

  deallocate_device<char>(&buffer);
}

template <typename T> void memoryLatency(int nwarp, int nsm) {

  T* bufferIn = NULL;
  T* bufferOut = NULL;
  allocate_device<T>(&bufferIn, 16384*nwarp*nsm);
  allocate_device<T>(&bufferOut, 16384*nwarp*nsm);

  cudaCheck(cudaDeviceSynchronize());

  // printf("%d\n", nwarp);
  // int nthread = 32*nwarp;
  // int nblock = nsm;
  // int shmemsize = nthread*sizeof(int);
  // memoryLatencyKernel<T> <<< nblock, nthread, shmemsize >>>(bufferIn, bufferOut);
  // cudaCheck(cudaGetLastError());

  int nthread = 32*nwarp;
  int nblock = nsm;
  int shmemsize = nthread*sizeof(int);
  memoryLatencyKernel2<T> <<< nblock, nthread, shmemsize >>>(bufferOut);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());
  // printf("\n");

  deallocate_device<T>(&bufferIn);
  deallocate_device<T>(&bufferOut);

}

void clearCache(int* buffer, const int bufferSize) {
  cudaCheck(cudaDeviceSynchronize());

  int nthread = 1024;
  int nblock = (bufferSize - 1)/nthread + 1;
  clearCacheKernel <<< nblock, nthread >>>(buffer, bufferSize);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());
}

void cyclesPerOperation() {
  cudaCheck(cudaDeviceSynchronize());

  int nthread = 32;
  int nblock = 1;
  cyclesPerOperationKernel <<< nblock, nthread >>>();
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());  
}

void printDeviceInfo() {
  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
  cudaSharedMemConfig pConfig;
  cudaCheck(cudaDeviceGetSharedMemConfig(&pConfig));
  int shMemBankSize = 4;
  if (pConfig == cudaSharedMemBankSizeEightByte) shMemBankSize = 8;
  double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  SM_major = prop.major;
  printf("Using %s SM version %d.%d\n", prop.name, prop.major, prop.minor);
  printf("Clock %1.3lfGhz numSM %d ECC %d mem BW %1.2lfGB/s shMemBankSize %dB\n", (double)prop.clockRate/1e6,
	 prop.multiProcessorCount, prop.ECCEnabled, mem_BW, shMemBankSize);
  printf("L2 %1.2lfMB\n", (double)prop.l2CacheSize/(double)(1024*1024));
}
