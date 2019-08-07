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

__global__ static void bag_set(ull *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    const ull id = blockIdx.x * blockDim.x + threadIdx.x;

	 ull i = id / set_size;
	 ull j = id % set_size;
	 bool b = id < bag_size * set_size && set[j] == bag[2 * i];
	 if (b) result[i] = 1;
}

__global__ static void join_scatter(ull const *il, ull const *ir, ull const *cl, ull const *cr, ull const *outpos,
                         ull  *l, ull *r, ull const x, ull const y, ull const z, ull const dump) {

    ull const id = blockIdx.x * blockDim.x + threadIdx.x;

    bool b = id < x * y * z;
    ull i = id / y / z * b;
    ull j = id / z % y;
    ull k = id % z;
    ull left = cl[i];
    ull right = cr[i];
    ull pos = outpos[i];

    b = b && !(j / left) && !(k / right);
    if (b) {
        left = pos + left * k + j;
        right = pos + right * j + k;

        l[left] = il[i] + j;
        r[right] = ir[i] + k;
    }
}
template<typename T>
__global__ static void parser(T *output, ull const *idx, unsigned char const *input, ull const rows, ull const loops) {
    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    bool b = id < rows;
    ull const s = idx[2 * id * b];
    ull const len = idx[(2 * id + 1) * b] - 1;
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
        bool d = c != '-' && j != dec;
        number += (len > 0 && d && i < len) * (c - '0') * (T)pow(10.0, p);
        p -= d;
    }
    output[b * id + !b * rows] = number * (!neg - neg);
}

__global__ static void string_gather(unsigned char *output, ull const *idx, unsigned char const *input, ull const size, ull const rows, ull const loops) {
    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    bool a = id < rows;
    ull const istart = idx[(3 * id) * a];
    ull const len = idx[(3 * id + 1) * a];
    ull const ostart = idx[(3 * id + 2) * a];
    bool b;

    for (ull i = 0; i < loops; ++i) {
        b = a && i < len;
        output[b * (ostart + i) + !b * (size - 1)] = b * input[istart + b * i];
    }
}

__global__ static void str_cmp(bool *output, unsigned char const *left, unsigned char const *right,
                               ull const *l_idx, ull const *r_idx, ull const rows, ull const loops) {
    ull const id = blockIdx.x * blockDim.x + threadIdx.x;
    bool a = id < rows;
    ull const l_start = l_idx[(2 * id) * a];
    ull const r_start = r_idx[(2 * id) * a];
    ull const len = l_idx[(2 * id + 1) * a];
    bool out = output[a * id + !a * rows];

    for (ull i = 0; i < loops; ++i) {
        out &= (len < i || left[l_start + i] == right[r_start + i]);
    }

    output[a * id + !a * rows] = out;
}

void inline bagSetIntersect(af::array &bag, af::array const &set) {
    using namespace af;
    auto const bag_size = bag.row(0).elements();
    auto const set_size = set.elements();
#ifdef USING_AF
    auto result = constant(0, dim4(1, bag_size + 1), u64);
    auto id = range(dim4(1, bag_size * set_size), 1, u64);
    auto i = id / set_size;
    auto j = id % set_size;
    auto b = moddims(set(j), i.dims()) == moddims(bag(0, i),i.dims());
    auto k = b * i + !b * bag_size;
    result(k) = 1;
    result = result.cols(0, end - 1);
#else
    auto result = constant(0, dim4(1, bag_size), u64);
    ull const threadLimit = THREAD_LIMIT;
    ull const threadCount = bag_size * set_size;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    auto set_ptr = set.device<ull>();
    auto result_ptr = result.device<ull>();
    auto bag_ptr = bag.device<ull>();
    af::sync();

    bag_set<<<grid, block>>>(result_ptr, bag_ptr, set_ptr, bag_size, set_size);

    bag.unlock();
    set.unlock();
    result.unlock();
#endif
    bag = bag(span, where(result));
    bag.eval();
}

void inline joinScatter(af::array &lhs, af::array &rhs, ull const equals) {
    using namespace af;
    using namespace TPCDI_Utils;
    Logger::startTimer("Join Prepare");
    auto left_count = accum(join(1, constant(1, 1, u64), (diff1(lhs.row(0), 1) > 0).as(u64)), 1) - 1;
    left_count = hflat(histogram(left_count, left_count.elements())).as(u64);
    left_count = left_count(left_count > 0);
    auto left_max = sum<unsigned int>(max(left_count, 1));
    auto left_idx = scan(left_count, 1, AF_BINARY_ADD, false);

    auto right_count = accum(join(1, constant(1, 1, u64), (diff1(rhs.row(0), 1) > 0).as(u64)), 1) - 1;
    right_count = hflat(histogram(right_count, right_count.elements())).as(u64);
    right_count = right_count(right_count > 0);
    auto right_max = sum<unsigned int>(max(right_count, 1));
    auto right_idx = scan(right_count, 1, AF_BINARY_ADD, false);

    auto output_pos = right_count * left_count;
    auto output_size = sum<ull>(output_pos);
    output_pos = scan(output_pos, 1, AF_BINARY_ADD, false);
    Logger::logTime("Join Prepare");
#ifdef USING_AF
    array left_out(1, output_size + 1, u64);
    array right_out(1, output_size + 1, u64);
    auto i = range(dim4(1, equals * left_max * right_max), 1, u64);
    auto j = i / right_max % left_max;
    auto k = i % right_max;
    i = i / left_max / right_max;
    auto b = !(j / left_count(i)) && !(k / right_count(i));
    left_out(b * (output_pos(i) + left_count(i) * k + j) + !b * output_size) = left_idx(i) + j;
    right_out(b * (output_pos(i) + right_count(i) * j + k) + !b * output_size) = right_idx(i) + k;
    left_out = left_out.cols(0, end - 1);
    right_out = right_out.cols(0, end - 1);
#else
    array left_out(1, output_size, u64);
    array right_out(1, output_size, u64);
    auto idx_l = left_idx.device<ull>();
    auto idx_r = right_idx.device<ull>();
    auto count_l = left_count.device<ull>();
    auto count_r = right_count.device<ull>();
    auto pos = output_pos.device<ull>();
    auto left = left_out.device<ull>();
    auto right = right_out.device<ull>();
    af::sync();
    ull const threadLimit = THREAD_LIMIT;
    ull const threadCount = equals * left_max * right_max;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);
    join_scatter<<<grid, block>>>(idx_l, idx_r, count_l, count_r, pos, left, right, equals, left_max, right_max, output_size);

    left_idx.unlock();
    right_idx.unlock();
    left_count.unlock();
    right_count.unlock();
    output_pos.unlock();
    left_out.unlock();
    right_out.unlock();
#endif
    lhs = lhs(1, left_out);
    rhs = rhs(1, right_out);
    lhs.eval();
    rhs.eval();
}

af::array inline stringGather(af::array const &input, af::array &indexer) {
    using namespace af;
    indexer = join(0, indexer, scan(indexer.row(1), 1, AF_BINARY_ADD, false));
    auto const loop_length = sum<ull>(max(indexer.row(1), 1));
    auto const out_length = sum<ull>(indexer.row(1));
    auto output = array(out_length, u8);

    #ifdef USING_AF
    for (ull i = 0; i < loop_length; ++i) {
        auto b = indexer.row(1) > i;
        af::array o_idx = indexer(2, b) + i;
        af::array i_idx = indexer(0, b) + i;
        output(o_idx) = (array)input(i_idx);
    }
    output.eval();
    #else
    auto const row_nums = indexer.elements() / 3;
    auto out_ptr = output.device<unsigned char>();
    auto in_ptr = input.device<unsigned char>();
    auto idx_ptr = indexer.device<ull>();
    af::sync();
    ull const threadLimit = THREAD_LIMIT;
    ull const threadCount = row_nums;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);
    string_gather<<<grid, block>>>(out_ptr, idx_ptr, in_ptr, out_length, row_nums, loop_length);
    output.unlock();
    input.unlock();
    indexer.unlock();
    #endif
    indexer.row(0) = (array)indexer.row(2);
    indexer = indexer.rows(0, 1);
    indexer.eval();
    return output;
}

af::array inline stringComp(af::array const &lhs, af::array const &rhs, af::array const &l_idx, af::array const &r_idx) {
    using namespace af;
    auto out = l_idx.row(1) == r_idx.row(1);
    auto loops = sum<ull>(max(l_idx(1, out)));
    #ifdef USING_AF
    for (ull i = 0; i < loops; ++i) {
        out = flat(out) && (flat(l_idx.row(1) < i) || flat(lhs(l_idx.row(0) + i) == rhs(r_idx.row(0) + i)) );
    }
    out.eval();
    #else
    auto out_ptr = (bool*)out.device<char>();
    auto left_ptr = lhs.device<unsigned char>();
    auto right_ptr = rhs.device<unsigned char>();
    auto l_idx_ptr = l_idx.device<ull>();
    auto r_idx_ptr = r_idx.device<ull>();
    auto const rows = l_idx.elements() / 2;
    af::sync();
    ull const threadLimit = THREAD_LIMIT;
    ull const threadCount = rows;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    str_cmp<<<grid, block>>>(out_ptr, left_ptr, right_ptr, l_idx_ptr, r_idx_ptr, rows, loops);
    out.unlock();
    lhs.unlock();
    rhs.unlock();
    l_idx.unlock();
    r_idx.unlock();
    #endif

    return out;
}

template<typename T>
af::array inline numericParse(af::array const &input, af::array const &indexer) {
    using namespace af;

    auto const loop_length = sum<ull>(max(indexer.row(1), 1));
    auto const row_nums = indexer.elements() / 2;
    auto output = array(1, row_nums + 1, GetAFType<T>().af_type);
    auto out_ptr = output.device<T>();
    auto idx_ptr = indexer.device<ull>();
    auto in_ptr = input.device<unsigned char>();
    af::sync();

    ull const threadLimit = THREAD_LIMIT;
    ull const threadCount = row_nums;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    parser<T><<<grid, block>>>(out_ptr, idx_ptr, in_ptr, row_nums, loop_length);

    output.unlock();
    input.unlock();
    indexer.unlock();
    output = output.cols(0, end - 1);
    output.eval();
    return output;
}
#endif
