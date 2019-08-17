#ifndef ARRAYFIRE_TPCDI_AFPARSER_H
#define ARRAYFIRE_TPCDI_AFPARSER_H

#include "include/Enums.h"
#include <arrayfire.h>

#ifndef ULL
    #define ULL
    typedef unsigned long long ull;
#endif

class Column;

class AFParser {
private:
    af::array _data = af::array(0, u8);
    af::array _indexer = af::array(0, u64);
    ull _length = 0;
    ull _width = 0;
    char _delimiter = 0;
    char const* _filename = nullptr;
    void _generateIndexer(bool hasHeader);
public:
    AFParser(char const *filename, char delimiter, bool hasHeader = false);
    AFParser(std::string const &text, char delimiter, bool hasHeader = false);
    AFParser(const std::vector<std::string> &files, char delimiter, bool hasHeader = false);
    AFParser() = default;
    virtual ~AFParser();
    template <typename T> Column parse(int column) const;
    Column asDate(int column, bool isDelimited = false, DateFormat inputFormat = YYYYMMDD) const;
    Column asDateTime(int column, DateFormat inputFormat = YYYYMMDD) const;
    Column asTime(int column, bool isDelimited) const;
};


#endif /* AFParser_hpp */
