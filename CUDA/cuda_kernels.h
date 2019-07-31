#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda.h>
#include <stdio.h>
#ifndef ULL
    #define ULL
typedef ull ull;
#endif
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

__global__
static void index_scatter(ull const *il, __global ull const *ir, ull const *cl, ull const *cr, ull const *outpos,
                     ull  *l, ull *r, ull const x, ull const y, ull const z, ull const dump) {

    ull const id = get_global_id(0);

    bool b = i < x * y * z;
    ull i = id / y / z * b;
    ull j = id / z % y;
    ull k = id % z;
    ull left = cl[i];
    ull right = cr[i];
    ull pos = outpos[i];

    b = b && !(j / left) && !(k / right);
    left = b * (pos + left * k + j) + !b * dump;
    right = b * (pos + right * j + k) + !b * dump;

    l[left] = il[i] + j;
    r[right] = ir[i] + k;
}

void inline launch_IndexScatter(ull const *il, ull const *ir, ull const *cl, ull const *cr, ull const *outpos,
        ull  *l, ull *r, ull const x, ull const y, ull const z, ull const dump) {
    ull const threadLimit = 1024;
    ull const threadCount = i_size * comp_size;
    ull const blocks = (threadCount/threadLimit) + 1;

    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    index_scatter<<<grid, block>>>((il, ir, cl, cr, outpos, l, r, x, y, z, dump);
}

#endif