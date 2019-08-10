#ifndef AFParser_hpp
#define AFParser_hpp

#include "include/Enums.h"
#include <arrayfire.h>

#ifndef ULL
    #define ULL
    typedef unsigned long long ull;
#endif

class Column;

class AFParser {
private:
    af::array _data;
    af::array _indexer;
    ull _length;
    ull _width;
    /* Excluding commas */
    ull* _maxColumnWidths;
    /* Excluding comma after the column */
    ull* _cumulativeMaxColumnWidths;
    char const* _filename;

    void _generateIndexer(char delimiter, bool hasHeader);
public:
    AFParser(char const *filename, char delimiter, bool hasHeader = false);
    AFParser(std::string const &text, char delimiter, bool hasHeader = false);
    AFParser(const std::vector<std::string> &files, char delimiter, bool hasHeader = false);
    AFParser() = default;
    virtual ~AFParser();
    void printData() const;
    template <typename T> Column parse(int column) const;
    Column asDate(int column, bool isDelimited = false, DateFormat inputFormat = YYYYMMDD) const;
    Column asDateTime(int column, DateFormat inputFormat = YYYYMMDD) const;
    Column asTime(int column, bool isDelimited) const;
};


#endif /* AFParser_hpp */
