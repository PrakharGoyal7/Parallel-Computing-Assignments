//Q1.
#include <stdio.h>
#include <cuda.h>
#define N 1024
__global__ void sumIterative(int *arr, int *result) {
    int tid = threadIdx.x;
    if (tid < N) {
        atomicAdd(result, arr[tid]);
    }
}
int main() {
    int h_input[N], h_result = 0;
    int *d_input, *d_result;
    for (int i = 0; i < N; i++)
        h_input[i] = i + 1;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    sumIterative<<<1, N>>>(d_input, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Iterative Sum: %d\n", h_result);
    cudaFree(d_input);
    cudaFree(d_result);
    return 0;
}


//Q2.
#include <stdio.h>
#define SIZE 1000
void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];
    for (i = 0; i < n1; i++) L[i] = arr[l + i];
    for (j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    i = j = 0; k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}
void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}
int main() {
    int arr[SIZE];
    for (int i = 0; i < SIZE; i++) arr[i] = rand() % 1000;
    mergeSort(arr, 0, SIZE - 1);
    printf("CPU Merge Sort Completed\n");
    return 0;
}


