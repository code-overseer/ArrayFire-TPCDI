#ifdef USING_CUDA

#include "Kernels.h"
#include "AFTypes.h"
#include "Utils.h"
#include <cuda.h>
#include <utility>
#include <cuda_profiler_api.h>

#define THREAD_LIMIT 1024
typedef unsigned long long ull;

__global__ static void cross_intersect(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {

    const ull i = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (i < bag_size) {
        ull const j = (ull)blockIdx.y;
        if (set[j] == bag[i]) result[i] = 1;
    }
}

__global__ static void hash_intersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
                               unsigned int const buckets, ull const bag_size) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;

    if (id < bag_size) {
        ull const val = bag[id];
        unsigned int const key = val % buckets;
        // send all keys within block to shared memory
        // sort keys in order
        // now ht_occ and ht_ptr will have coalesced memory access
        unsigned int const len = ht_occ[key];
        ull const ptr = ht_ptr[key];
        // send all lengths within block to shared memory
        // send all ptrs within block to shared memory
        // sort ptrs and lengths in order of lengths
        // now divergence will be minimized, work efficiency increase
        char out = 0;
        for (uint i = 0; i < len; ++i) {
            out |= (ht_val[ptr + i] == val);
        }
        result[id] = out;
    }
}

__global__ static void join_scatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                         ull  *l, ull *r, ull const equals, ull const l_max, ull const r_max) {

    ull const i = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (i < equals) {
        ull const j = (ull)blockIdx.y;
        ull const k = (ull)blockIdx.z;
        ull const left = l_cnt[i];

        if (j < left && k < r_cnt[i]) {
            ull const pos = outpos[i] + left * k + j;
            l[pos] = l_idx[i] + j;
            r[pos] = r_idx[i] + k;
        }
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

__global__ static void string_gather(unsigned char *output, ull const *idx, unsigned char const *input, ull const rows) {

    ull const r = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (r < rows) {
        ull const istart = idx[3 * r];
        ull const len = idx[3 * r + 1] - 1;
        ull const ostart = idx[3 * r + 2];
        memcpy(&output[ostart], &input[istart], len);
        output[ostart + len] = 0;
    }
}

__global__ static void str_cmp(bool *output, unsigned char const *left, unsigned char const *right,
                               ull const *l_idx, ull const *r_idx, unsigned int const * mask, ull const rows) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (id < rows) {
        unsigned int i = mask[id];

        ull const l_start = l_idx[2 * i];
        ull const len = l_idx[2 * i + 1];
        ull const r_start = r_idx[2 * i];

        bool out = 1;

        for (long long j = 0; j < len; ++j) {
            out &= left[l_start + j] == right[r_start + j];
        }

        output[i] = out;
    }
}

__global__ static void str_cmp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
        ull const rows, ull const loops) {
    ull const id = (ull)blockIdx.x * (ull)blockDim.x + (ull)threadIdx.x;
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        bool out = output[id];

        for (int i = 0; i < loops; ++i) {
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
        ull const l_end = left_idx[2 * j + 1] - 1;
        ull const r_start = right_idx[2 * j];
        ull const r_end = right_idx[2 * j + 1] - 1;
        ull const o_start = out_idx[2 * j];
        ull const o_end = out_idx[2 * j + 1] - 1;

        bool const b = k < o_end;
        bool const c = k < l_end;
        bool const d = c ^ b;
        i = b * k + !b * o_end;
        j = c * k + !c * l_end;
        k = d * (k - l_end) + !b * r_end;
        output[o_start + i] = c * left[l_start + j] + d * right[r_start + k];
    }
}

static std::pair<ull, int> blockFinder(ull const size) {
    ull out = (size >> 5u) + ((size & ((1u << 5u) - 1)) > 0);
    for (unsigned int i = 6; i < 11; ++i) {
        ull tmp = (size >> i) + ((size & ((1u << i) - 1)) > 0);
        if (tmp == out) return { out, 1u << (i - 1) };
        else out = tmp;
    }
    return { out, THREAD_LIMIT };
}

void launchCrossIntersect(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    auto layout = blockFinder(bag_size);
    dim3 grid(layout.first, set_size, 1);
    dim3 block(layout.second, 1, 1);
    cudaProfilerStart();
    cross_intersect<<<grid, block>>>(result, bag, set, bag_size, set_size);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void launchHashIntersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
                                unsigned int const buckets, ull const bag_size) {
    auto layout = blockFinder(bag_size);
    dim3 grid(layout.first, 1, 1);
    dim3 block(layout.second, 1, 1);
    cudaProfilerStart();
    hash_intersect<<<grid, block>>>(result, bag, ht_val, ht_ptr, ht_occ, buckets, bag_size);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                      ull *left, ull *right, ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    auto layout = blockFinder(equals);
    ull const y = left_max;
    ull const z = right_max;
    dim3 grid(layout.first, left_max, right_max);
    dim3 block(layout.second, 1, 1);

    cudaProfilerStart();
    join_scatter<<<grid, block>>>(l_idx, r_idx, l_cnt, r_cnt, outpos, left, right, equals, left_max, right_max);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

template<typename T>
void launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows) {
    auto layout = blockFinder(rows);
    dim3 grid(layout.first, 1, 1);
    dim3 block(layout.second, 1, 1);

    cudaProfilerStart();
    parser<T><<<grid, block>>>(output, idx, input, rows);
    cudaDeviceSynchronize();
    cudaProfilerStop();

}
#define PARSER(TYPE) \
template void launchNumericParse<TYPE>(TYPE *output, ull const * idx, unsigned char const *input, ull const rows);

PARSER(unsigned char)
PARSER(float)
PARSER(double)
PARSER(unsigned short)
PARSER(short)
PARSER(unsigned int)
PARSER(int)
PARSER(ull)
PARSER(long long)

#undef PARSER

void launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const output_size,
        ull const rows, ull const loops) {
    auto layout = blockFinder(rows);

    dim3 grid(layout.first, 1, 1);
    dim3 block(layout.second, 1, 1);

    cudaProfilerStart();
    string_gather<<<grid, block>>>(output, idx, input, rows);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                      ull const *l_idx, ull const *r_idx, unsigned int const *mask, ull const rows) {
    auto layout = blockFinder(rows);
    dim3 grid(layout.first, 1, 1);
    dim3 block(layout.second, 1, 1);

    cudaProfilerStart();
    str_cmp<<<grid, block>>>(output, left, right, l_idx, r_idx, mask, rows);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                      ull const rows, ull const loops) {
    auto layout = blockFinder(rows);
    dim3 grid(layout.first, 1, 1);
    dim3 block(layout.second, 1, 1);

    cudaProfilerStart();
    str_cmp<<<grid, block>>>(output, left, right, l_idx, rows, loops);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

#endif
