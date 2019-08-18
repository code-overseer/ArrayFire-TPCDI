#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include "Kernels.h"
#include "AFTypes.h"
#include "Utils.h"
#include <arrayfire.h>
#include <cuda.h>

#define THREAD_LIMIT 1024
typedef unsigned long long ull;

__global__ static void bag_set(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {

    const ull id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;

	 ull i = id / set_size;
	 ull j = id % set_size;
	 bool b = id < bag_size * set_size;
	 if (b && set[j] == bag[i * b]) result[i] = 1;
}

__global__ static void hash_intersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
                               unsigned int const buckets, ull const bag_size) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;

    if (id < bag_size) {
        ull const val = bag[id];
        unsigned int const key = val % buckets;
        unsigned int const len = ht_occ[key];
        ull const ptr = ht_ptr[key];

        char out = 0;
        for (unsigned int i = 0; i < len; ++i) {
            out |= (ht_val[ptr + i] == val);
        }
        result[id] = out;
    }
}

__global__ static void join_scatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                         ull  *l, ull *r, ull const equals, ull const l_max, ull const r_max, ull const dump) {

    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;

    bool b = id < equals * l_max * r_max;
    ull i = id / l_max / r_max * b;
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
__global__ static void parser(T *output, ull const *idx, unsigned char const *input, ull const rows) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (id < rows) {
        long long const s = idx[2 * id];
        long long const len = idx[2 * id + 1] - 1;
        T number = 0;
        unsigned char dec = 0;
        bool frac = 0;
        bool neg = input[s] == '-';
        for (long long i = 0; i < len; ++i) {
            unsigned char digit = input[s + i];
            bool b = digit >= '0' && digit <= '9';
            frac |= digit == '.';
            dec += b && frac;
            bool c = !dec && b;
            number = number * (c * 10 + !c) + b * (digit - '0') / (T)pow(10.0, dec);
        }

        output[id] = number * (!neg - neg);
    }
}

__global__ static void string_gather(unsigned char *output, ull const *idx, unsigned char const *input, ull const size, ull const rows, ull const loops) {

    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
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
                               ull const *l_idx, ull const *r_idx, ull const rows) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (id < rows) {
        ull const l_start = l_idx[2 * id];
        ull const r_start = r_idx[2 * id];
        ull const len = l_idx[2 * id + 1];
        bool out = output[id];

        for (long long i = 0; i < len; ++i) {
            out &= (len < i || left[l_start + i] == right[r_start + i]);
        }

        output[id] = out;
    }
}

__global__ static void str_cmp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
        ull const rows, ull const loops) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        bool out = output[id];

        for (long long i = 0; i < loops; ++i) {
            out &= left[l_start + i] == right[i];
        }
        output[id] = out;
    }
}

__global__ static void str_concat(unsigned char *output, ull const *out_idx, unsigned char const *left,
ull const *left_idx,  unsigned char const *right,  ull const *right_idx, ull const out_length,
ull const rows, ulong const loops) {

    ull i = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    ull j = i / loops;
    ull k = i % loops;
    if (j < rows) {
        ull const l_start = left_idx[2 * j];
        ull const r_start = right_idx[2 * j];
        ull const o_start = out_idx[2 * j];
        ull const l_end = left_idx[2 * j + 1] - 1;
        ull const o_end = out_idx[2 * j + 1] - 1;
        ull const r_end = right_idx[2 * j + 1] - 1;

        bool const b = k < o_end;
        bool const c = k < l_end;
        bool const d = c ^ b;
        i = b * k + !b * o_end;
        j = c * k + !c * l_end;
        k = d * (k - l_end) + !b * r_end;
        output[o_start + i] = c * left[l_start + j] + d * right[r_start + k];
    }
}

void launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    ull const threadCount = bag_size * set_size;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    bag_set<<<grid, block>>>(result, bag, set, bag_size, set_size);
    cudaDeviceSynchronize();
}

void launchHashIntersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
                                unsigned int const buckets, ull const bag_size) {
    ull const threadCount = bag_size;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    hash_intersect<<<grid, block>>>(result, bag, ht_val, ht_ptr, ht_occ, buckets, bag_size);
    cudaDeviceSynchronize();
}

void lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                      ull *left, ull *right, ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    ull const threadCount = equals * left_max * right_max;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    join_scatter<<<grid, block>>>(l_idx, r_idx, l_cnt, r_cnt, outpos, left, right, equals, left_max, right_max, out_size);
    cudaDeviceSynchronize();
}

template<typename T>
void launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows) {
    ull const threadCount = rows;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);

    parser<T><<<grid, block>>>(output, idx, input, rows);
    cudaDeviceSynchronize();

}

void launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const output_size,
        ull const rows, ull const loops) {
    ull const threadCount = rows * loops;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);
    string_gather<<<grid, block>>>(output, idx, input, output_size, rows, loops);
    cudaDeviceSynchronize();
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                      ull const *l_idx, ull const *r_idx, ull const rows) {
    ull const threadCount = rows;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);

    str_cmp<<<grid, block>>>(output, left, right, l_idx, r_idx, rows);
    cudaDeviceSynchronize();
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                      ull const rows, ull const loops) {
    ull const threadCount = rows;
    ull const blocks = (threadCount/THREAD_LIMIT) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(THREAD_LIMIT, 1, 1);

    str_cmp<<<grid, block>>>(output, left, right, l_idx, rows, loops);
    cudaDeviceSynchronize();
}

#endif