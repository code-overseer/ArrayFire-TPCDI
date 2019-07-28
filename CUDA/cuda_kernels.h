#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda.h>
#include <stdio.h>

__global__
static void IsExists(uint64_t *result, uint64_t const *input, uint64_t const *comparison, uint64_t const i_size, uint64_t const comp_size) {
    const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;

	 uint64_t i = id / comp_size;
	 uint64_t j = id % comp_size;
	 bool b = id < i_size * comp_size && comparison[j] == input[2 * i];
	 uint64_t k = b * i + !b * i_size;

	 result[k] = 1;
}

void inline launch_IsExist(uint64_t *result, uint64_t const *input, uint64_t const *comparison, uint64_t const i_size, uint64_t const comp_size) {
    uint64_t const threadLimit = 1024;
    uint64_t const threadCount = i_size * comp_size;
    uint64_t const blocks = (threadCount/threadLimit) + 1;

    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    IsExists<<<grid, block>>>(result, input, comparison, i_size, comp_size);
}

#endif