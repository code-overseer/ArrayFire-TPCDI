#ifndef AFParser_hpp
#define AFParser_hpp

#include <arrayfire.h>
#include <unordered_map>
#include <iostream>
#include "Enums.h"
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
#endif

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
    af::array _makeUniform(int column) const;
    af::array _numParse(int column, af::dtype type) const;
public:
    AFParser(char const *filename, char delimiter, bool hasHeader = false);
    AFParser(std::string const &text, char delimiter, bool hasHeader = false);
    AFParser(const std::vector<std::string> &files, char delimiter, bool hasHeader = false);
    AFParser() = default;
    virtual ~AFParser();
    void printData() const;
    af::array asBoolean(int column) const;
    af::array asDate(int column, bool isDelimited = false, DateFormat inputFormat = YYYYMMDD) const;
    af::array asDateTime(int column, bool isDelimited = false, DateFormat inputFormat = YYYYMMDD) const;
    af::array asUchar(int column) const;
    af::array asUshort(int column) const;
    af::array asShort(int column) const;
    af::array asUint(int column) const;
    af::array asInt(int column) const;
    af::array asFloat(int column) const;
    af::array asU64(int column) const;
    af::array asS64(int column) const;
    af::array asDouble(int column) const;
    af::array asString(int column) const;
    af::array asTime(int column) const;

    af::array asFloat2(int column) const;
    af::array asString2(int i) const;
};


#endif /* AFParser_hpp */
