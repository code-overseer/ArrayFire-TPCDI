#include "KernelInterface.h"
#include "Kernels.h"
#include "AFHashTable.h"
#include "AFTypes.h"
#include "Utils.h"
#include <cstring>
#include <Logger.h>

typedef unsigned long long ull;

af::array crossIntersect(af::array const &bag, af::array const &set) {
    using namespace af;
    Logger::startTimer("NL Bag Set");
    auto const bag_size = bag.row(0).elements();
    auto const set_size = set.elements();
#ifdef USING_AF
    auto result = constant(0, dim4(1, bag_size), b8);
    auto id = range(dim4(1, bag_size * set_size), 1, u64);
    auto i = id / set_size;
    auto j = id % set_size;
    auto b = moddims(set(j), i.dims()) == moddims(bag(0, i), i.dims());
    result(i(b) + 0) = 1;
#else
    auto result = constant(0, dim4(1, bag_size), b8);
    auto result_ptr = result.device<char>();
    auto set_ptr = set.device<ull>();
    auto bag_ptr = bag.row(0).device<ull>();
    af::sync();

    launchCrossIntersect(result_ptr, bag_ptr, set_ptr, bag_size, set_size);

    bag.unlock();
    set.unlock();
    result.unlock();
#endif
    af::array out = bag(span, result);
    Logger::logTime("NL Bag Set", false);
    return out;
}

af::array hashIntersect(af::array const &bag, AFHashTable const &ht) {
    using namespace af;
    using namespace Utils;
    Logger::startTimer("Hash Bag Set");
    auto const bag_size = bag.row(0).elements();
    #ifdef USING_AF
    auto len = sum<ull>(max(ht.getOcc(), 1));
    auto result = constant(0, dim4(1, bag_size), b8);
    auto keys = bag.row(0) % ht.buckets();
    for (ull i = 0; i < len; ++i) {
        auto idx = ht.getOcc(keys) > 0 && i < ht.getOcc(keys);
        af::array k = keys(idx);
        result(idx) = result(idx) || (ht.getValues(ht.getPtr(k) + i) == bag(0, idx));
    }
    result.eval();
    #else
    auto result = constant(0, dim4(1, bag_size), b8);
    auto result_ptr = result.device<char>();
    auto bag_ptr = bag.row(0).device<ull>();
    af::sync();

    if (bag_size > 64) launchHashIntersect(result_ptr, bag_ptr, ht.values(), ht.pointers(), ht.occupancy(), ht.buckets(), bag_size);
    else launchCrossIntersect(result_ptr, bag_ptr, ht.values(), bag_size, ht.getValues().elements());

    bag.unlock();
    ht.unlock();
    result.unlock();
    #endif

    af::array out = bag(span, result);
    out.eval();
    Logger::logTime("Hash Bag Set", false);

    return out;
}

void joinScatter(af::array &lhs, af::array &rhs, ull const equals) {
    using namespace af;
    using namespace Utils;

    Logger::startTimer("Join Scatter");

    auto diffe = diff1(lhs.row(0), 1) > 0;

    auto left_idx = hflat(where64(join(1, af::constant(1,1,diffe.type()), diffe)));
    auto left_count = hflat(where64(join(1, diffe, af::constant(1,1,diffe.type())))) - left_idx + 1; // histogram
    auto left_max = sum<unsigned int>(max(left_count, 1));

    diffe = diff1(rhs.row(0), 1) > 0;
    auto right_idx = hflat(where64(join(1, af::constant(1,1,diffe.type()), diffe)));
    auto right_count = hflat(where64(join(1, diffe, af::constant(1,1,diffe.type())))) - right_idx + 1; // histogram
    auto right_max = sum<unsigned int>(max(right_count, 1));

    auto output_pos = right_count * left_count;
    auto output_size = sum<ull>(output_pos);
    output_pos = (output_pos.elements() == 1) ? constant(0, 1, output_pos.type())
                                              : scan(output_pos, 1, AF_BINARY_ADD, false);
#ifdef USING_AF
    array left_out(1, output_size, u64);
    array right_out(1, output_size, u64);
    auto i = range(dim4(1, equals * left_max * right_max), 1, u64);
    auto j = i / right_max % left_max;
    auto k = i % right_max;
    i = i / left_max / right_max;
    auto b = (j < left_count(i)) && (k < right_count(i));
    auto idx = output_pos(i) + left_count(i) * k + j;
    idx = idx(b);
    i = i(b);
    left_out(idx) = left_idx(i) + j(b);
    right_out(idx) = right_idx(i) + k(b);
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
    Logger::logTime("Join Scatter", false);
}

af::array stringGather(af::array const &input, af::array &indexer) {
    using namespace af;
    Logger::startTimer("String Gather");
    indexer = join(0, indexer, indexer.elements() < 3 ?
                               constant(0, 1, indexer.type()) :
                               scan(indexer.row(1), 1, AF_BINARY_ADD, false));

    indexer.eval();

    auto const out_size = sum<ull>(indexer(seq(1, 2), end));
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
    indexer = join(0, indexer.row(2), indexer.row(1));
    indexer.eval();

    Logger::logTime("String Gather", false);
    return output;
}

af::array stringComp(af::array const &lhs, af::array const &rhs, af::array const &l_idx, af::array const &r_idx) {
    using namespace af;
    Logger::startTimer("String Comparison");
    if (l_idx.elements() != r_idx.elements()) throw std::runtime_error("Expected columns with same length");
    auto out = l_idx.row(1) == r_idx.row(1);
    #ifdef USING_AF
    auto loops = sum<ull>(max(l_idx(1, out)));
    for (ull i = 0; i < loops; ++i) {
        out(out) = out(out) && Utils::hflat(lhs(l_idx(0, out) + i) == rhs(r_idx(0, out) + i));
    }
    #else
    auto mask_idx = where(out);
    auto out_ptr = (bool*) out.device<char>();
    auto left_ptr = lhs.device<unsigned char>();
    auto right_ptr = rhs.device<unsigned char>();
    auto l_idx_ptr = l_idx.device<ull>();
    auto r_idx_ptr = r_idx.device<ull>();
    auto mask_idx_ptr = mask_idx.device<unsigned int>();
    auto const rows = mask_idx.elements();
    af::sync();

    launchStringComp(out_ptr, left_ptr, right_ptr, l_idx_ptr, r_idx_ptr, mask_idx_ptr, rows);

    out.unlock();
    lhs.unlock();
    rhs.unlock();
    l_idx.unlock();
    r_idx.unlock();
    #endif
    out.eval();
    Logger::logTime("String Comparison", false);
    return out;
}

af::array stringComp(af::array const &lhs, char const *rhs, af::array const &l_idx) {
    using namespace af;
    Logger::startTimer("String Comparison");
    auto loops = strlen(rhs) + 1;
    auto out = l_idx.row(1) == loops;
    #ifdef USING_AF
    for (ull i = 0; i < loops; ++i) {
        out(out) = out(out) && Utils::hflat(lhs(l_idx(0, out) + i) == rhs[i]);
    }
    #else
    auto right = array(loops, rhs).as(u8);
    auto out_ptr = (bool *) out.device<char>();
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
    Logger::logTime("String Comparison", false);
    return out;
}

template<typename T>
af::array numericParse(af::array const &input, af::array const &indexer) {
    using namespace af;
    using namespace Utils;
    Logger::startTimer("Numeric Parse");
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

    digit_idx = array();
    len = array();
    dec = array();
    neg = array();
    callGC();
    #else
    auto out_ptr = output.template device<T>();
    auto idx_ptr = indexer.device<ull>();
    auto in_ptr = input.device<unsigned char>();
    af::sync();
    launchNumericParse<T>(out_ptr, idx_ptr, in_ptr, rows);
    output.unlock();
    input.unlock();
    indexer.unlock();
    output.eval();
    #endif
    Logger::logTime("Numeric Parse", false);
    return output;
}

#define PARSER(TYPE) \
template af::array numericParse<TYPE>(af::array const &input, af::array const &indexer);

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
