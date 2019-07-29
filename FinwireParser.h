//
// Created by Bryan Wong on 2019-07-03.
//

#ifndef ARRAYFIRE_TPCDI_FINWIREPARSER_H
#define ARRAYFIRE_TPCDI_FINWIREPARSER_H

#include "AFParser.hpp"
#include "AFDataFrame.h"
#include <functional>
#include <string>

struct Finwire {
public:
    AFDataFrame company;
    AFDataFrame financial;
    AFDataFrame security;
    Finwire(AFDataFrame&& cmp, AFDataFrame&& fin, AFDataFrame&& sec) : company(cmp), financial(fin), security(sec) {}
    void clear() { company.clear(); financial.clear(); security.clear(); }
};

class FinwireParser {
private:
    enum RecordType { FIN = 0, CMP = 1, SEC = 2 };
    AFDataFrame extractData(RecordType type) const;
    AFDataFrame extractCmp() const;
    AFDataFrame extractFin() const;
    AFDataFrame extractSec() const;
    static af::array _PTSToDatetime(af::array &PTS, bool isDelimited = false, DateFormat inputFormat = YYYYMMDD);
    char const _search[3][4] = {"FIN", "CMP", "SEC"};
    int const _widths[3] = {17, 16, 12};
    int const _FINLengths[18] = {15, 3, 4, 1, 8, 8, 17, 17, 12, 12, 12, 17, 17, 17, 13, 13, 60, -1};
    int const _CMPLengths[17] = {15, 3, 60, 10, 4, 2, 4, 8, 80, 80, 12, 25, 20, 24, 46, 150, -1};
    int const _SECLengths[13] = {15, 3, 15, 6, 4, 70, 6, 13, 8, 8, 12, 60, -1};
    af::array _finwireData;
    af::array _indexer;
    af::array _extract(af::array const &start, int length) const;
    uint32_t _maxRowWidth;
public:
    explicit FinwireParser(std::vector<std::string> const &files);
    Finwire extractData() const { return Finwire(extractCmp(), extractFin(), extractSec()); }
};
#endif //ARRAYFIRE_TPCDI_FINWIREPARSER_H
