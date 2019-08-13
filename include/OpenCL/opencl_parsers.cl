#ifndef PARSE_TYPE
#define PARSE_TYPE float
#endif
__kernel void parser(__global PARSE_TYPE *output, __global ulong const *idx, __global uchar const *input,
        ulong const row_num) {
    ulong const id = get_global_id(0);
    if (id < row_num) {
        ulong const s = idx[2 * id];
        ulong const len = idx[2 * id + 1] - 1;
        PARSE_TYPE number = 0;
        uint dec = 0;
        // two loops is slightly slower but, better precision parsing
        #pragma unroll LOOP_LENGTH
        for (ulong i = 0; i < LOOP_LENGTH; ++i) {
            ulong j = i * (i < len);
            dec += (input[s + j] == '.') * j;
        }
        bool neg = input[s] == '-';

        int p = (int)(dec + !dec * len) - 1 - neg;

        #pragma  unroll LOOP_LENGTH
        for (ulong i = 0; i < LOOP_LENGTH; ++i) {
            ulong j = i * (i < len);
            uchar c = input[s + j];
            bool d = c != '-' && c != '.';
            number += (len > 0 && d && i < len) * (c - '0') * (PARSE_TYPE)pown(10.0, p);
            p -= d;
        }
        output[id] = number * (!neg - neg);
    }
}

__kernel void str_cmp(__global bool *output, __global uchar const *left, __global uchar const *right,
        __global ulong const *l_idx, __global ulong const *r_idx, ulong const rows) {

    ulong const id = get_global_id(0);
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        ulong const len = l_idx[2 * id + 1];
        ulong const r_start = r_idx[2 * id];
        bool out = output[id];
        #pragma unroll LOOP_LENGTH
        for (ulong i = 0; i < LOOP_LENGTH; ++i) {
            out &= (len < i || left[l_start + i] == right[r_start + i]);
        }
        output[id] = out;
    }
}

__kernel void str_cmp_single(__global bool *output, __global uchar const *left, __global uchar const *right,
__global ulong const *l_idx, ulong const rows) {

    ulong const id = get_global_id(0);
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        bool out = output[id];
        #pragma unroll LOOP_LENGTH
        for (ulong i = 0; i < LOOP_LENGTH; ++i) {
            out &= left[l_start + i] == right[i];
        }
        output[id] = out;
    }
}