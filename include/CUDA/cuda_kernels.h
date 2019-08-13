#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include "include/Logger.h"
#include <cuda.h>
#include <stdio.h>
#include <arrayfire.h>
#include "include/AFTypes.h"
#include "include/TPCDI_Utils.h"
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
#endif

#define THREAD_LIMIT 1024;

__global__ static void bag_set(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {

    const ull id = blockIdx.x * blockDim.x + threadIdx.x;

	 ull i = id / set_size;
	 ull j = id % set_size;

	 if (id < bag_size * set_size && set[j] == bag[2 * i]) result[i] = 1;
}

__global__ static void join_scatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                         ull  *l, ull *r, ull const equals, ull const l_max, ull const r_max, ull const dump) {

    ull const id = blockIdx.x * blockDim.x + threadIdx.x;

    bool b = id < equals * l_max * r_max;
    ull i = id / l_max / r_maxz * b;
    ull j = id / r_max % l_max;
    ull k = id % r_max;
    ull left = l_cnt[i];
    ull right = r_cnt[i];
    ull pos = outpos[i];
    
    if (b && !(j / left) && !(k / right)) {
        left = pos + left * k + j;
        l[left] = l_idx[i] + j;
        r[left] = r_idx[i] + k;
    }
}
template<typename T>
__global__ static void parser(T *output, ull const *idx, unsigned char const *input, ull const rows, ull const loops) {
    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < rows) {
        ull const s = idx[2 * id];
        ull const len = idx[2 * id + 1] - 1;
        T number = 0;
        ull dec = 0;
        // Multiple accesses to input[s+j] inefficient
        for (ull i = 0; i < loops; ++i) {
            ull j = i * (i < len);
            dec += (input[s + j] == '.') * j;
        }

        bool neg = input[s] == '-';
        int p = (int)(dec + !dec * len) - 1 - neg;

        for (ull i = 0; i < loops; ++i) {
            ull j = i * (i < len);
            unsigned char c = input[s + j];
            bool d = c != '-' && c != '.';
            number += (len > 0 && d && i < len) * (c - '0') * (T)pow(10.0, p);
            p -= d;
        }

        output[id] = number * (!neg - neg);
    }
}

__global__ static void string_gather(unsigned char *output, ull const *idx, unsigned char const *input, ull const size, ull const rows, ull const loops) {

    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    ull const r = id / loops;
    ull const l = id % loops;
    if (r < rows) {
        ull const istart = idx[3 * r];
        ull const len = idx[3 * r + 1];
        ull const ostart = idx[3 * r + 2];
        bool b = l < len;
        bool c = l != len - 1;
        output[b * (ostart + l) + !b * (size - 1)] = b * input[istart + b * l] * c;
    }
}

__global__ static void str_cmp(bool *output, unsigned char const *left, unsigned char const *right,
                               ull const *l_idx, ull const *r_idx, ull const rows, ull const loops) {
    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < rows) {
        ull const l_start = l_idx[2 * id];
        ull const r_start = r_idx[2 * id];
        ull const len = l_idx[2 * id + 1];
        bool out = output[id];

        for (ull i = 0; i < loops; ++i) {
            out &= (len < i || left[l_start + i] == right[r_start + i]);
        }

        output[id] = out;
    }
}

__global__ static void str_cmp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
        ull const rows, ull const loops) {
    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        bool out = output[id];

        for (ulong i = 0; i < loops; ++i) {
            out &= left[l_start + i] == right[i];
        }
        output[id] = out;
    }
}

__global__ static void str_concat(unsigned char *output, ull const *out_idx, unsigned char const *left,
ull const *left_idx,  unsigned char const *right,  ull const *right_idx, ull const out_length,
ull const rows, ulong const loops) {

    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    ull const r = id / loops;
    ull const l = id % loops;
    if (r < rows) {
        ull const l_start = left_idx[2 * id];
        ull const r_start = right_idx[2 * id];
        ull const o_start = out_idx[2 * id];
        ull const l_len = left_idx[2 * id + 1] - 1;
        ull const o_len = out_idx[2 * id + 1];

        bool b = l < l_len;
        bool c = (l < o_len) ^ b;
        bool d = l != o_len - 1;

        output[c * (o_start + l) + !c * (size - 1)] = (b * left[l_start + b * l] + c * right[r_start + c * l]) * d;
    }
}

void inline launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    ull const threadCount = bag_size * set_size;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    bag_set<<<grid, block>>>(result, bag, set, bag_size, set_size);
}

void inline lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                      ull *left, ull *right, ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    ull const threadCount = equals * left_max * right_max;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    join_scatter<<<grid, block>>>(l_idx, r_idx, l_cnt, r_cnt, outpos, left, right, equals, left_max, right_max, out_size);
}

template<typename T>
void inline launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows, ull const loops) {
    ull const threadCount = rows;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);

    parser<T><<<grid, block>>>(out_ptr, idx_ptr, in_ptr, rows, loops);
}

void inline launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const output_size,
        ull const rows, ull const loops) {
    ull const threadCount = rows * loops;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    string_gather<<<grid, block>>>(out_ptr, idx_ptr, in_ptr, output_size, rows, loops);
}

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                      ull const *l_idx, ull const *r_idx, ull const rows, ull const loops) {
    ull const threadCount = rows;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);

    str_cmp<<<grid, block>>>(output, left, right, l_idx, r_idx, rows, loops);
}

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                      ull const rows, ull const loops) {
    ull const threadCount = rows;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);

    str_cmp<<<grid, block>>>(out_ptr, left_ptr, right_ptr, l_idx_ptr, rows, loops);
}

#endif
