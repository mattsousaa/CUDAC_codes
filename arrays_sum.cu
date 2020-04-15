#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>

#define CHECK(call){                                                        \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess){                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
    /*
    block_offset = blockIdx.x * blockDim.x
    row_offset   = gridDim.x * blockDim.x * blockIdx.y

    1D grid - index = threadIdx.x + block_offset = threadIdx.x + blockIdx.x * blockDim.x 
    2D grid - index = threadIdx.x + row_offset   = threadIdx.x + gridDim.x * blockDim.x * blockIdx.y
    */
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) C[i] = A[i] + B[i];  
}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx = 0; idx < N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d \n",hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

void initialData(float *v_A, float *v_B, int size){
// generate different seed for random number
time_t t;

srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++) {
        v_A[i] = (float)( rand() & 0xFF )/10.0f;
        v_B[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

int main(int argc, char **argv){

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

//int nElem = atoi(argv[1]);  // Size of vectors
int nElem = 1<<24;
printf("Vector size %d\n", nElem);

size_t nBytes = nElem * sizeof(float);  // Size in bytes of each vector

float *h_A, *h_B;       // Host input vectors
float *hostRef;         // Host output vector
float *gpuRef;          // Device output vector for tests

float *d_A, *d_B;       // Device input vectors
float *d_C;             // Device output vector

// Allocate memory for each vector on host
h_A = (float *)malloc(nBytes);
h_B = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef = (float *)malloc(nBytes);

double iStart, iElaps;

// Initialize vectors on host
initialData(h_A, h_B, nElem);

memset(hostRef, 0, nBytes);
memset(gpuRef, 0, nBytes);

// Allocate memory for each vector on GPU
cudaMalloc(&d_A, nBytes);
cudaMalloc(&d_B, nBytes);
cudaMalloc(&d_C, nBytes);

// Copy host vectors to device
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

// invoke kernel at host side
int iLen = 1024;
dim3 block (iLen);
dim3 grid ((nElem+block.x-1)/block.x);

iStart = cpuSecond();
// Execute the kernel
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
cudaDeviceSynchronize();
iElaps = cpuSecond() - iStart;
printf("sumArraysOnGPU <<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

// Copy array back to host
cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

// add vector at host side for result checks
sumArraysOnHost(h_A, h_B, hostRef, nElem);

// check device results
checkResult(hostRef, gpuRef, nElem);

// Release device memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

// Release host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);

}