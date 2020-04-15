#include <stdio.h>

#define CHECK(call){                                                        \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess){                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny);
double cpuSecond();
void checkResult(float *hostRef, float *gpuRef, const int N);
void initialData(float *v_A, float *v_B, int size);