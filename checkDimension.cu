#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
            "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
            gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc, char **argv) {
            
int nX = 16;
int nY = 16;    // define total data element
//int nEle = 16;
//int iLen = 8;

// define grid and block structure
dim3 block (8, 8);
dim3 grid (nX/block.x, nY/block.y);
//dim3 block (iLen, iLen);
//dim3 grid ((nEle + block.x-1)/block.x, (nEle + block.y-1)/block.y);

// check grid and block dimension from host side
printf("grid.x %d grid.y %d grid.z %d\n",grid.x, grid.y, grid.z);
printf("block.x %d block.y %d block.z %d\n",block.x, block.y, block.z);

// check grid and block dimension from device side
checkIndex <<<grid, block>>> ();

// reset device before you leave
cudaDeviceReset();
return(0);
}