//
// Created by Bryan Wong on 2019-07-03.
//

#ifndef ARRAYFIRE_TPCDI_FINWIREPARSER_H
#define ARRAYFIRE_TPCDI_FINWIREPARSER_H

#include "AFParser.hpp"
#include "AFDataFrame.h"
#include <functional>
#include <string>

class FinwireParser {
public:
    std::function<std::string(char const*)> loadFile = AFParser::loadFile;

    FinwireParser(const std::vector<std::string> &files);
    enum RecordType { FIN = 0, CMP = 1, SEC = 2 };
    AFDataFrame extractData(RecordType type) const;
    AFDataFrame extractCmp() const;
    AFDataFrame extractFin() const;
    AFDataFrame extractSec() const;
    static af::array stringToNum(af::array &numstr, af::dtype type);
    static af::array stringToDate(af::array &datestr, DateFormat inputFormat = YYYYMMDD, bool isDelimited = false);
    static af::array stringToTime(af::array &timestr, bool isDelimited = false);
private:
    static af::array _PTSToDatetime(af::array &PTS, DateFormat inputFormat = YYYYMMDD, bool isDelimited = false);
    static std::string collect(const std::vector<std::string> &files);
    char const _search[3][4] = {"FIN", "CMP", "SEC"};
    int const _widths[3] = {17, 16, 12};
    int const _FINLengths[18] = {15, 3, 4, 1, 8, 8, 17, 17, 12, 12, 12, 17, 17, 17, 13, 13, 60, -1};
    int const _CMPLengths[17] = {15, 3, 60, 10, 4, 2, 4, 8, 80, 80, 12, 25, 20, 24, 46, 150, -1};
    int const _SECLengths[13] = {15, 3, 15, 6, 4, 70, 6, 13, 8, 8, 12, 60, -1};
    af::array _finwireData;
    af::array _indexer;
    af::array _extract(af::array const &start, int length) const;
    uint32_t _maxRowWidth;

};
#endif //ARRAYFIRE_TPCDI_FINWIREPARSER_H
