//
// Created by Bryan Wong on 2019-06-28.
//

#include "TPC_DI.h"

AFDataFrame loadDimDate(char const *filepath) {
    AFDataFrame frame;
    {
        auto parser = std::make_unique<AFParser>(filepath, '|');
        frame.add(parser->asUlong(0), AFDataFrame::ULONG);
        frame.add(parser->asDate(1, YYYYMMDD, true), AFDataFrame::DATE);
        for (int i = 2;  i < 17; i += 2) {
            frame.add(parser->asString(i), AFDataFrame::STRING);
            frame.add(parser->asUint(i + 1), AFDataFrame::UINT);
        }
        frame.add(parser->stringToBoolean(17), AFDataFrame::BOOL);
    }
    return frame;
}

AFDataFrame loadDimBroker(char const* filepath, AFDataFrame& dimDate) {
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
    auto length = frame.data()[0].dims(0);
    frame.insert(af::range(af::dim4(length), 0, u64), AFDataFrame::ULONG, 0);
    frame.add(af::constant(1, af::dim4(length), b8), AFDataFrame::BOOL);
    dimDate.dateSort(1);

    frame.add(af::tile(dimDate.data()[1](0,af::span), length), AFDataFrame::DATE);
    frame.add(af::tile(AFDataFrame::endDate(), length), AFDataFrame::DATE);
    return frame;
}