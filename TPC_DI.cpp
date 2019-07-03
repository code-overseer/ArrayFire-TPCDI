//
// Created by Bryan Wong on 2019-06-28.
//

#include "TPC_DI.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace fs = boost::filesystem;

AFDataFrame loadDimDate(char const *filepath) {
    AFDataFrame frame;
    AFParser parser(filepath, '|', false);
    frame.add(parser.asUlong(0), AFDataFrame::ULONG);
    frame.add(parser.asDate(1, YYYYMMDD, true), AFDataFrame::DATE);
    for (int i = 2;  i < 17; i += 2) {
        frame.add(parser.asString(i), AFDataFrame::STRING);
        frame.add(parser.asUint(i + 1), AFDataFrame::UINT);
    }
    frame.add(parser.stringToBoolean(17), AFDataFrame::BOOL);
    return frame;
}

AFDataFrame loadDimTime(char const* filepath) {
    AFDataFrame frame;
    AFParser parser(filepath, '|', false);
    frame.add(parser.asUlong(0), AFDataFrame::ULONG);
    frame.add(parser.asTime(1), AFDataFrame::TIME);
    for (int i = 2;  i < 7; i += 2) {
        frame.add(parser.asUint(i), AFDataFrame::UINT);
        frame.add(parser.asString(i + 1), AFDataFrame::STRING);
    }
    frame.add(parser.stringToBoolean(8), AFDataFrame::BOOL);
    frame.add(parser.stringToBoolean(9), AFDataFrame::BOOL);
    return frame;
}

AFDataFrame loadIndustry(char const* filepath) {
    AFDataFrame frame;
    AFParser parser(filepath, '|', false);
    for (int i = 0;  i < 3; ++i) {
        frame.add(parser.asString(i), AFDataFrame::STRING);
    }
    return frame;
}

AFDataFrame loadStatusType(char const* filepath) {
    AFDataFrame frame;
    AFParser parser(filepath, '|', false);
    for (int i = 0;  i < 2; ++i) {
        frame.add(parser.asString(i), AFDataFrame::STRING);
    }
    return frame;
}

AFDataFrame loadTaxRate(char const* filepath) {
    AFDataFrame frame;
    AFParser parser(filepath, '|', false);
    for (int i = 0;  i < 2; ++i) {
        frame.add(parser.asString(i), AFDataFrame::STRING);
    }
    frame.add(parser.asFloat(2), AFDataFrame::FLOAT);
    return frame;
}

AFDataFrame loadTradeType(char const* filepath) {
    AFDataFrame frame;
    AFParser parser(filepath, '|', false);
    for (int i = 0;  i < 2; ++i) {
        frame.add(parser.asString(i), AFDataFrame::STRING);
    }
    for (int i = 2;  i < 4; ++i) {
        frame.add(parser.asUint(i), AFDataFrame::UINT);
    }
    return frame;
}

AFDataFrame loadAudit(char const* folderpath) {
    af::setBackend(AF_BACKEND_CPU);
    AFDataFrame frame;
    std::string path(folderpath);
    std::vector< std::string > auditFiles;
    fs::directory_iterator end_itr; // Default ctor yields past-the-end
    for( fs::directory_iterator i( folderpath ); i != end_itr; ++i )
    {
        auto n = i->path().string().find("_audit.csv", path.size());
        if( n == std::string::npos) continue;
        auditFiles.push_back( i->path().string() );
    }
    for (int i = 0; i < auditFiles.size(); i++) {

        AFParser parser(auditFiles[i].c_str(), ',', true);
        if (!i) {
            frame.add(parser.asString(0), AFDataFrame::STRING);
            frame.add(parser.asUchar(1), AFDataFrame::UINT);
            frame.add(parser.asDate(2, YYYYMMDD, true), AFDataFrame::DATE);
            frame.add(parser.asString(3), AFDataFrame::STRING);
            frame.add(parser.asInt(4), AFDataFrame::LONG);
            frame.add(parser.asFloat(5), AFDataFrame::DOUBLE);
        } else {
            AFDataFrame tmp;
            tmp.add(parser.asString(0), AFDataFrame::STRING);
            tmp.add(parser.asUchar(1), AFDataFrame::UINT);
            tmp.add(parser.asDate(2, YYYYMMDD, true), AFDataFrame::DATE);
            tmp.add(parser.asString(3), AFDataFrame::STRING);
            tmp.add(parser.asInt(4), AFDataFrame::LONG);
            tmp.add(parser.asFloat(5), AFDataFrame::DOUBLE);
            frame.concatenate(tmp);
        }

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