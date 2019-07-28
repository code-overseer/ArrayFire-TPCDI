__kernel void is_exist_kernel(__global ulong *result, __global ulong const *input,
        __global ulong const *comparison, ulong const i_size, ulong const comp_size) {

    const ulong id = get_global_id(0);

    ulong i = id / comp_size;
    ulong j = id % comp_size;
    bool b = id < i_size * comp_size && comparison[j] == input[2 * i];
    ulong k = b * i + !b * i_size;

    result[k] = 1;
}