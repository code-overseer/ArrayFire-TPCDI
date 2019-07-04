#include "Tests.h"
#include "AFDataFrame.h"

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
    af_print(result)
    af::sync();
}

void test_stringToBool(char const *filepath) {
    auto test = AFParser(filepath, '|', false);
    auto result = test.stringToBoolean(17);
    af_print(result)
    af::sync();
}

void test_Date(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asDate(0, YYYYMMDD);
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
        frame.add(lval, AFDataFrame::UINT);
        lval = parser->asUint(1);
        frame.add(lval, AFDataFrame::UINT);
        for (int i = 2; i <= 7; ++i) {
            lval = parser->asString(i);
            frame.add(lval, AFDataFrame::STRING);
        }
    }

    frame.stringMatch(5, "314");
    frame.remove(5);
    auto lval = af::range(af::dim4(frame.data()[0].dims(0)), 0, u64);
    frame.insert(lval, AFDataFrame::U64, 0);
    lval = af::constant(1, af::dim4(frame.data()[0].dims(0)), b8);
    frame.add(lval, AFDataFrame::BOOL);
    lval = af::constant(19500101, af::dim4(frame.data()[0].dims(0)), u32);
    frame.add(lval, AFDataFrame::DATE);
    lval = af::constant(99991231, af::dim4(frame.data()[0].dims(0)), u32);
    frame.add(lval, AFDataFrame::DATE);
}
