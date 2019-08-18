#include "include/Tests.h"
#include "include/Utils.h"
#include "include/KernelInterface.h"
#ifndef ULL
#define ULL
    typedef unsigned long long ull;
#endif
using namespace Utils;

void test_SignedInt(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<int>(0);
//    result.printColumn();
    af::sync();
}

void test_UnsignedInt(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<unsigned int>(0);
    result.printColumn();
    af::sync();
}

void test_UChar(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<unsigned char>(0);
    result.printColumn();
    af::sync();
}

void test_Float(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<float>(0);
//    result.printColumn();
    af::sync();
}

void test_Double(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<double>(0);
    result.printColumn();
    af::sync();
}

void test_UnsignedLong(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<unsigned long long>(0);
    result.printColumn();
    af::sync();
}

void test_SignedLong(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<long long>(0);
    result.printColumn();
    af::sync();
}

void test_String(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.parse<char*>(0);
    result.printColumn();
    af::sync();
}

void test_stringToBool(char const *filepath) {
    auto test = AFParser(filepath, '|', false);
    auto result = test.parse<bool>(17);
    result.printColumn();
    af::sync();
}

void test_Date(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asDate(0, true);
    result.printColumn();
    af::sync();
}

void test_Time(char const *filepath) {
    auto test = AFParser(filepath, '|', false);
    auto result = test.asTime(1, false);
    result.printColumn();
    af::sync();
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
        lhs = hflat(lhs);
        lhs = join(0, lhs, range(lhs.dims(), 1, u64));
        rhs = hflat(rhs);
        rhs = join(0, rhs, range(rhs.dims(), 1, u64));
    }
    auto equalSet = hflat(setIntersect(setUnique(lhs.row(0)), setUnique(rhs.row(0)), true));

    bagSetIntersect(lhs, equalSet);
    bagSetIntersect(rhs, equalSet);

    auto equals = equalSet.elements();

    joinScatter(lhs, rhs, equals);

    af_print(lhs);
}


