#include <stdio.h>

__global__ void helloFromGPU(void){
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv){
    helloFromGPU <<<1, 10>>>();
    cudaDeviceSynchronize();

    return 0;
}

