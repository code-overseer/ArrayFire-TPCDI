#include "include/Tests.h"
#include "include/AFDataFrame.h"
#include <memory>
#include "include/TPCDI_Utils.h"
#if defined(USING_OPENCL)
#include "OpenCL/opencl_kernels.h"
#elif defined(USING_CUDA)
#include "CUDA/cuda_kernels.h"
#else
#include "CPU/vector_functions.h"
#endif
#ifndef ULL
#define ULL
    typedef unsigned long long ull;
#endif
using namespace TPCDI_Utils;

void test_SignedInt(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asInt(0);
    af_print(result)
    af::sync();
}

void test_UnsignedInt(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asUint(0);
    af_print(result)
    af::sync();
}

void test_UChar(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asUchar(0);
    af_print(result)
    af::sync();
}

void test_Float(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asFloat(0);
    af_print(result)
    af::sync();
}

void test_Double(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asDouble(0);
    af_print(result)
    af::sync();
}

void test_UnsignedLong(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asU64(0);
    af_print(result)
    af::sync();
}

void test_SignedLong(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asS64(0);
    af_print(result)
    af::sync();
}

void test_String(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asString(0);
//    af_print(result)
    stringToNum(result, s32);
    af::sync();
}

void test_stringToBool(char const *filepath) {
    auto test = AFParser(filepath, '|', false);
    auto result = test.asBoolean(17);
    af_print(result)
    af::sync();
}

void test_Date(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asDate(0, true);
    af_print(result)
    af::sync();
}

void test_Time(char const *filepath) {
    auto test = AFParser(filepath, '|', false);
    auto result = test.asTime(1);
    af_print(result)
    af::sync();
}

void test_StringMatch(char const *filepath) {
    AFDataFrame frame;
    {
        auto parser = std::make_unique<AFParser>(filepath, ',');
        auto lval = parser->asUint(0);
        frame.add(lval, UINT);
        lval = parser->asUint(1);
        frame.add(lval, UINT);
        for (int i = 2; i <= 7; ++i) {
            lval = parser->asString(i);
            frame.add(lval, STRING);
        }
    }

    auto idx = frame.stringMatch(5, "314");
    frame = frame.select(idx);
    frame.remove(5);
    auto lval = af::range(af::dim4(frame.data()[0].dims(0)), 0, u64);
    frame.insert(lval, ULONG, 0);
    lval = af::constant(1, af::dim4(frame.data()[0].dims(0)), b8);
    frame.add(lval, BOOL);
    lval = af::constant(19500101, af::dim4(frame.data()[0].dims(0)), u32);
    frame.add(lval, DATE);
    lval = af::constant(99991231, af::dim4(frame.data()[0].dims(0)), u32);
    frame.add(lval, DATE);
}

void hashTest(char const *filepath) {
    
    AFDataFrame frame;
    {
        AFParser file(filepath,',');
        frame.add(file.asString(0), STRING, "Words");
        af::sync();
    }

    af::timer::start();
    auto h = polyHash(byteHash(frame.data("Words")));
    af::sync();
    printf("%f\n", af::timer::stop());
    af::array idx;
    af::sort(h, idx, flipdims(h));
    auto i = af::setUnique(h, true);
    auto j = diff1(h);
    j = where(j == 0);
    j = join(0, j, j + 1);
    j.eval();
    auto im = (i % 375761llU);
    frame = frame.select(idx);
    if (j.isempty()) return;
}

void testSetJoin() {
    using namespace af;
    array lhs;
    array rhs;
    {
        int l[] = {2,3,3,5,5,6,6,6,6,8,8,9,9,9};
        int r[] = {2,3,4,4,5,5,5,6,7,7,7,9,9,11,12};
        lhs = array(14, l).as(u64);
        rhs = array(15, r).as(u64);
        lhs = flipdims(lhs);
        lhs = join(0, lhs, range(lhs.dims(), 1, u64));
        rhs = flipdims(rhs);
        rhs = join(0, rhs, range(rhs.dims(), 1, u64));
    }
    auto equalSet = flipdims(setIntersect(setUnique(lhs.row(0)), setUnique(rhs.row(0)), true));

    bagSetIntersect(lhs, equalSet);
    bagSetIntersect(rhs, equalSet);

    auto equals = equalSet.elements();

    joinScatter(lhs, rhs, equals);

    af_print(lhs);
}


