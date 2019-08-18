//
// Created by Bryan Wong on 2019-07-03.
//

#ifndef ARRAYFIRE_TPCDI_FINWIREPARSER_H
#define ARRAYFIRE_TPCDI_FINWIREPARSER_H

#include "AFParser.h"
#include "AFDataFrame.h"
#include <functional>
#include <string>

struct Finwire {
public:
    AFDataFrame company;
    AFDataFrame financial;
    AFDataFrame security;

    Finwire(AFDataFrame &&cmp, AFDataFrame &&fin, AFDataFrame &&sec) : company(cmp), financial(fin), security(sec) {}

    void clear() {
        company.clear();
        financial.clear();
        security.clear();
    }
};

class FinwireParser {
private:
    enum RecordType {
        FIN = 0, CMP = 1, SEC = 2
    };
    char const _search[3][4] = {"FIN", "CMP", "SEC"};
    ull const _FINLengths[18] = {15, 3, 4, 1, 8, 8, 17, 17, 12, 12, 12, 17, 17, 17, 13, 13, 60, 0};
    ull const _CMPLengths[17] = {15, 3, 60, 10, 4, 2, 4, 8, 80, 80, 12, 25, 20, 24, 46, 150, 0};
    ull const _SECLengths[13] = {15, 3, 15, 6, 4, 70, 6, 13, 8, 8, 12, 60, 0};
    af::array _data;
    af::array _indexer;

    Column _extract(af::array &start, unsigned int length, RecordType const &type) const;

public:
    AFDataFrame extractCmp() const;

    AFDataFrame extractFin() const;

    AFDataFrame extractSec() const;

    explicit FinwireParser(std::vector<std::string> const &files);

    virtual ~FinwireParser() {
        _data = af::array(0, u8);
        _indexer = af::array(0, u64);
        af::deviceGC();
    }

    inline Finwire extractData() const { return Finwire(extractCmp(), extractFin(), extractSec()); }

    af::array filterRowsByCategory(const RecordType &type) const;
};

#endif //ARRAYFIRE_TPCDI_FINWIREPARSER_H
