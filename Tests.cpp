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
    auto result = test.asUlong(0);
    af_print(result)
    af::sync();
}

void test_SignedLong(char const *filepath) {
    auto test = AFParser(filepath, ',', false);
    auto result = test.asLong(0);
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
    auto result = test.asDate(0, YYYYMMDD, true);
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
        frame.add(parser->asUint(0), AFDataFrame::UINT);
        frame.add(parser->asUint(1), AFDataFrame::UINT);
        frame.add(parser->asString(2), AFDataFrame::STRING);
        frame.add(parser->asString(3), AFDataFrame::STRING);
        frame.add(parser->asString(4), AFDataFrame::STRING);
        frame.add(parser->asString(5), AFDataFrame::STRING);
        frame.add(parser->asString(6), AFDataFrame::STRING);
        frame.add(parser->asString(7), AFDataFrame::STRING);
    }

    frame.stringMatch(5, "314");
    frame.remove(5);
    frame.insert(af::range(af::dim4(frame.data()[0].dims(0)), 0, u64), AFDataFrame::ULONG, 0);
    frame.add(af::constant(1, af::dim4(frame.data()[0].dims(0)), b8), AFDataFrame::BOOL);
    frame.add(af::constant(19500101, af::dim4(frame.data()[0].dims(0)), u32), AFDataFrame::DATE);
    frame.add(af::constant(99991231, af::dim4(frame.data()[0].dims(0)), u32), AFDataFrame::DATE);
}
