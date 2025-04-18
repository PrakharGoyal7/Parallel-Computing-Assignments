//Q1 (i)
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024
__device__ __constant__ float A[N];
__device__ __constant__ float B[N];
__device__ float C[N];
__global__ void vectorAddKernel() {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

//Q1 (ii)
int main() {
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    cudaMemcpyToSymbol(A, h_A, sizeof(float) * N);
    cudaMemcpyToSymbol(B, h_B, sizeof(float) * N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAddKernel<<<(N+255)/256, 256>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpyFromSymbol(h_C, C, sizeof(float) * N);
    printf("Kernel execution time: %.6f ms\n", ms);
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %.2f\n", i, h_C[i]);
    }
    
//Q1 (iii)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float memClock = prop.memoryClockRate / 1000.0f; 
    float memBusWidth = prop.memoryBusWidth;         
    float theoreticalBW = 2.0f * memClock * memBusWidth / 8 / 1000; // in GB/s
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoreticalBW);

//Q1 (iv)
    measuredBW=(RBytes+WBytes)/t;
    int bytesRead = 2 * N * sizeof(float);
    int bytesWritten = N * sizeof(float);
    float measuredBW = (bytesRead + bytesWritten) / (ms / 1000.0f) / 1e9;
    printf("Measured Bandwidth: %.2f GB/s\n", measuredBW);
    return 0;
}


