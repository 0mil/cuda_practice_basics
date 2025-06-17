#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

int main() {
    hello_cuda<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}

