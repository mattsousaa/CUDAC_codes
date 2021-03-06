#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>

#include "functions.h"
#include "kernel_files.h"

int main(int argc, char **argv){

printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up date size of matrix
int nx = 1<<14;
int ny = 1<<14;

int nxy = nx*ny;
int nBytes = nxy * sizeof(float);  // Size in bytes of matrix
printf("Matrix size: nx %d ny %d\n",nx, ny);

// malloc host memory
float *h_A, *h_B, *hostRef, *gpuRef;
h_A = (float *)malloc(nBytes);
h_B = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef = (float *)malloc(nBytes);

// initialize data at host side
double iStart = cpuSecond();
initialData (h_A, h_B, nxy);
double iElaps = cpuSecond() - iStart;

memset(hostRef, 0, nBytes);
memset(gpuRef, 0, nBytes);

// add matrix at host side for result checks
iStart = cpuSecond();
sumMatrixOnHost (h_A, h_B, hostRef, nx,ny);
iElaps = cpuSecond() - iStart;
printf("sumMatrixOnHOST elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_MatA, *d_MatB, *d_MatC;
cudaMalloc((void **)&d_MatA, nBytes);
cudaMalloc((void **)&d_MatB, nBytes);
cudaMalloc((void **)&d_MatC, nBytes);

// transfer data from host to device
cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

// invoke kernel at host side
/*
// ########## 2D x 2D ##########
int dimx = 32;
int dimy = 32;
dim3 block(dimx, dimy);
dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
*/


// ########## 1D x 1D ##########
int dimx = 32;
dim3 block(dimx);
dim3 grid((nx+block.x-1)/block.x);


/*
// ########## 2D x 1D ##########
int dimx = 32;
dim3 block(dimx);
dim3 grid((nx + block.x - 1) / block.x, ny);
*/

iStart = cpuSecond();
//sumMatrixOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
sumMatrixOnGPU1D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
//sumMatrixOnGPUMix <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
cudaDeviceSynchronize();
iElaps = cpuSecond() - iStart;
printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

// copy kernel result back to host side
cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

// check device results
checkResult(hostRef, gpuRef, nxy);

// free device global memory
cudaFree(d_MatA);
cudaFree(d_MatB);
cudaFree(d_MatC);

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

// reset device
cudaDeviceReset();
return (0);

}