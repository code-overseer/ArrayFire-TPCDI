#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda.h>
#include <stdio.h>
typedef unsigned long long ull;
__global__
static void IsExists(ull *result, ull const *input, ull const *comparison, ull const i_size, ull const comp_size) {
    const ull id = blockIdx.x * blockDim.x + threadIdx.x;

	 ull i = id / comp_size;
	 ull j = id % comp_size;
	 bool b = id < i_size * comp_size && comparison[j] == input[2 * i];
	 ull k = b * i + !b * i_size;

	 result[k] = 1;
}

void inline launch_IsExist(ull *result, ull const *input, ull const *comparison, ull const i_size, ull const comp_size) {
    ull const threadLimit = 1024;
    ull const threadCount = i_size * comp_size;
    ull const blocks = (threadCount/threadLimit) + 1;

    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    IsExists<<<grid, block>>>(result, input, comparison, i_size, comp_size);
}

#endif