#ifndef ARRAYFIRE_TPCDI_KERNELINTERFACE_H
#define ARRAYFIRE_TPCDI_KERNELINTERFACE_H
#include <arrayfire.h>
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
#endif
#include "AFTypes.h"
#if defined(USING_OPENCL)
#include "include/OpenCL/opencl_kernels.h"
#elif defined(USING_CUDA)
#include "include/CUDA/cuda_kernels.h"
#else
#include "include/CPU/single_threaded.h"
#endif

void inline bagSetIntersect(af::array &bag, af::array const &set) {
    using namespace af;
    auto const bag_size = bag.row(0).elements();
    auto const set_size = set.elements();
#ifdef USING_AF
    auto result = constant(0, dim4(1, bag_size + 1), b8);
    auto id = range(dim4(1, bag_size * set_size), 1, u64);
    auto i = id / set_size;
    auto j = id % set_size;
    auto b = moddims(set(j), i.dims()) == moddims(bag(0, i), i.dims());
    auto k = b * i + !b * bag_size;
    result(k) = 1;
    result = result.cols(0, end - 1);
#else
    auto result = constant(0, dim4(1, bag_size), b8);
    auto result_ptr = result.device<char>();
    auto set_ptr = set.device<ull>();
    auto bag_ptr = bag.device<ull>();
    af::sync();

    launchBagSet(result_ptr, bag_ptr, set_ptr, bag_size, set_size);

    bag.unlock();
    set.unlock();
    result.unlock();
#endif
    bag = bag(span, result);
    bag.eval();
}

void inline joinScatter(af::array &lhs, af::array &rhs, ull const equals) {
    using namespace af;
    using namespace TPCDI_Utils;
    auto left_count = accum(join(1, constant(1, 1, u64), (diff1(lhs.row(0), 1) > 0).as(u64)), 1) - 1;
    left_count = hflat(histogram(left_count, left_count.elements())).as(u64);
    left_count = left_count(left_count > 0);
    auto left_max = sum<unsigned int>(max(left_count, 1));
    auto left_idx = (left_count.elements() == 1) ? constant(0, 1, left_count.type())
                                                 : scan(left_count, 1, AF_BINARY_ADD, false);

    auto right_count = accum(join(1, constant(1, 1, u64), (diff1(rhs.row(0), 1) > 0).as(u64)), 1) - 1;
    right_count = hflat(histogram(right_count, right_count.elements())).as(u64);
    right_count = right_count(right_count > 0);
    auto right_max = sum<unsigned int>(max(right_count, 1));
    auto right_idx = (right_count.elements() == 1) ? constant(0, 1, right_count.type())
                                                   : scan(right_count, 1, AF_BINARY_ADD, false);

    auto output_pos = right_count * left_count;
    auto output_size = sum<ull>(output_pos);
    output_pos = (output_pos.elements() == 1) ? constant(0, 1, output_pos.type())
                                              : scan(output_pos, 1, AF_BINARY_ADD, false);
#ifdef USING_AF
    array left_out(1, output_size + 1, u64);
    array right_out(1, output_size + 1, u64);
    auto i = range(dim4(1, equals * left_max * right_max), 1, u64);
    auto j = i / right_max % left_max;
    auto k = i % right_max;
    i = i / left_max / right_max;
    auto b = !(j / left_count(i)) && !(k / right_count(i));
    auto idx = b * (output_pos(i) + left_count(i) * k + j) + !b * output_size;
    left_out(idx) = left_idx(i) + j;
    right_out(idx) = right_idx(i) + k;
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

    lauchJoinScatter(idx_l, idx_r, count_l, count_r, pos, left, right, equals, left_max, right_max, output_size);

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
    indexer = join(0, indexer, indexer.elements() < 3 ?
    constant(0, 1, indexer.type()) :
    scan(indexer.row(1), 1, AF_BINARY_ADD, false));

    indexer.eval();
    auto const out_size = sum<ull>(indexer.row(1));
    auto const loops = sum<ull>(max(indexer.row(1), 1));
    auto const rows = indexer.elements() / 3;
    auto output = array(out_size, u8);
    #ifdef USING_AF
    for (ull i = 0; i < loops; ++i) {
        auto b = indexer.row(1) > i;
        auto c = indexer(1, b) - 1 != i;
        output(indexer(2, b) + i) = input(indexer(0, b) + i) * flat(c);
    }
    output.eval();
    #else
    auto out_ptr = output.device<unsigned char>();
    auto in_ptr = input.device<unsigned char>();
    auto idx_ptr = indexer.device<ull>();
    af::sync();

    launchStringGather(out_ptr, idx_ptr, in_ptr, out_size, rows, loops);

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
        out(out) = out(out) && TPCDI_Utils::hflat(lhs(l_idx(0, out) + i) == rhs(r_idx(0, out) + i));
    }
    #else
    auto out_ptr = (bool*)out.device<char>();
    auto left_ptr = lhs.device<unsigned char>();
    auto right_ptr = rhs.device<unsigned char>();
    auto l_idx_ptr = l_idx.device<ull>();
    auto r_idx_ptr = r_idx.device<ull>();
    auto const rows = l_idx.elements() / 2;
    af::sync();

    launchStringComp(out_ptr, left_ptr, right_ptr, l_idx_ptr, r_idx_ptr, rows, loops);

    out.unlock();
    lhs.unlock();
    rhs.unlock();
    l_idx.unlock();
    r_idx.unlock();
    #endif
    out.eval();

    return out;
}

af::array inline stringComp(af::array const &lhs, char const *rhs, af::array const &l_idx) {
    using namespace af;
    auto loops = strlen(rhs) + 1;
    auto out = l_idx.row(1) == loops;
    #ifdef USING_AF
    for (ull i = 0; i < loops; ++i) {
        out(out) = out(out) && TPCDI_Utils::hflat(lhs(l_idx(0, out) + i) == rhs[i]);
    }
    #else
    auto right = array(loops, rhs).as(u8);
    auto out_ptr = (bool*)out.device<char>();
    auto left_ptr = lhs.device<unsigned char>();
    auto right_ptr = right.device<unsigned char>();
    auto l_idx_ptr = l_idx.device<ull>();
    auto const rows = l_idx.elements() / 2;
    af::sync();

    launchStringComp(out_ptr, left_ptr, right_ptr, l_idx_ptr, rows, loops);

    out.unlock();
    lhs.unlock();
    l_idx.unlock();
    right.unlock();
    #endif

    out.eval();

    return out;
}

template<typename T> af::array inline numericParse(af::array const &input, af::array const &indexer) {
    using namespace af;
    using namespace TPCDI_Utils;
    auto const loops = sum<ull>(max(indexer.row(1), 1)) - 1;
    auto const rows = indexer.elements() / 2;
    auto output = constant(0, dim4(1, rows), GetAFType<T>().af_type);
    if (!loops) return output;
    #ifdef USING_AF
    auto dec = constant(0, output.dims(), u8);
    auto frac = constant(0, output.dims(), b8);
    auto neg = frac;
    auto digit_idx = indexer.row(0) + 0;
    auto len = indexer.row(1) - 1;
    neg(input(digit_idx) == '-') = 1;
    for (int i = 0; i < loops; ++i) {
        auto j = i < len && len > 0;
        digit_idx(j) = digit_idx(j) + (i > 0);
        auto b = hflat(input(digit_idx) >= '0' && input(digit_idx) <= '9') && j;
        frac = frac || hflat(input(digit_idx) == '.');
        dec += (b && frac);
        output = output * pow(10, (!dec && b)) + b * hflat(input(digit_idx) - '0') / pow(10, dec.as(output.type()));
    }
    output = output * (!neg - neg);
    output.eval();

    dec = array();
    digit_idx = array();
    neg = array();
    len = array();
    deviceGC();
    #else
    auto out_ptr = output.template device<T>();
    auto idx_ptr = indexer.device<ull>();
    auto in_ptr = input.device<unsigned char>();
    af::sync();
    launchNumericParse<T>(out_ptr, idx_ptr, in_ptr, rows, loops);
    output.unlock();
    input.unlock();
    indexer.unlock();
    output.eval();
    #endif
    return output;
}

#endif //ARRAYFIRE_TPCDI_KERNELINTERFACE_H
