__kernel void parser(__global PARSE_TYPE *output, __global ulong const *idx, __global uchar const *input,
        ulong const row_num) {
    ulong const id = get_global_id(0);
    bool b = id < row_num;
    ulong const s = idx[2 * id * b];
    ulong const len = idx[(2 * id + 1) * b] - 1;
    PARSE_TYPE number = 0;
    ulong dec = 0;
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
        number += (len > 0 && c != '-' && j != dec && i < len) * (c - '0') * (PARSE_TYPE)pown(10.0, p);
        p -= (j != dec && c != '-');
    }
    output[b * id + !b * row_num] = number * (!neg - neg);
}