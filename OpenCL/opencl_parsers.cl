__kernel void parser(__global PARSE_TYPE *output, __global ulong const *idx, __global uchar const *input,
        ulong const row_num) {
    ulong const id = get_global_id(0);
    bool b = id < row_num;
    ulong const s = idx[2 * id * b];
    ulong const len = idx[(2 * id + 1) * b] - 1;
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
        bool d = c != '-' && j != dec;
        number += (len > 0 && d && i < len) * (c - '0') * (PARSE_TYPE)pown(10.0, p);
        p -= d;
    }
    output[b * id + !b * row_num] = number * (!neg - neg);
}

__kernel void string_gather(__global uchar *output, __global ulong const *idx, __global uchar const *input,
    ulong const out_length, ulong const row_num) {
    ulong const id = get_global_id(0);
    bool a = id < row_num;
    ulong const istart = idx[(3 * id) * a];
    ulong const len = idx[(3 * id + 1) * a];
    ulong const ostart = idx[(3 * id + 2) * a];
    bool b;

    #pragma unroll LOOP_LENGTH
    for (ulong i = 0; i < LOOP_LENGTH; ++i) {
        b = a && i < len;
        output[b * (ostart + i) + !b * (out_length - 1)] = b * input[istart + b * i];
    }
}