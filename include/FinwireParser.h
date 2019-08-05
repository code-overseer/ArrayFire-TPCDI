//
// Created by Bryan Wong on 2019-07-03.
//

#ifndef ARRAYFIRE_TPCDI_FINWIREPARSER_H
#define ARRAYFIRE_TPCDI_FINWIREPARSER_H

#include "AFParser.hpp"
#include "AFDataFrame.h"
#include <functional>
#include <string>
#ifndef ULL
#define ULL
    typedef unsigned long long ull;
#endif

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
    ull const _FINLengths[18] = {15llU, 3llU, 4llU, 1llU, 8llU, 8llU, 17llU, 17llU, 12llU, 12llU, 12llU, 17llU, 17llU, 17llU, 13llU, 13llU, 60llU, 0};
    ull const _CMPLengths[17] = {15llU, 3llU, 60llU, 10llU, 4llU, 2llU, 4llU, 8llU, 80llU, 80llU, 12llU, 25llU, 20llU, 24llU, 46llU, 150llU, 0};
    ull const _SECLengths[13] = {15llU, 3llU, 15llU, 6llU, 4llU, 70llU, 6llU, 13llU, 8llU, 8llU, 12llU, 60llU, 0};
    af::array _finwireData;
    af::array _indexer;
    af::array _extract(af::array const &start, const unsigned int length) const;
    uint32_t _maxRowWidth;
public:
    explicit FinwireParser(std::vector<std::string> const &files);
    Finwire extractData() const { return Finwire(extractCmp(), extractFin(), extractSec()); }
};
#endif //ARRAYFIRE_TPCDI_FINWIREPARSER_H
