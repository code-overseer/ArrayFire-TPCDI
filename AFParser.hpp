#ifndef AFParser_hpp
#define AFParser_hpp

#include <arrayfire.h>
#include <unordered_map>
#include <iostream>
#include "Enums.h"

/* For debug */
template<typename T>
void print(T i) {std::cout << i << std::endl;}

class AFParser {
private:
    af::array _data;
    af::array _indexer;
    unsigned long _length;
    unsigned long _width;
    /* Excluding commas */
    uint32_t* _maxColumnWidths;
    /* Excluding comma after the column */
    uint32_t* _cumulativeMaxColumnWidths;
    char const* _filename;

    void _generateIndexer(char delimiter, bool hasHeader);
    af::array _generateReversedCharacterIndices(int column) const;
    void _makeUniform(int column, af::array &output, af::array &negatives, af::array &points) const;
    af::array _numParse(int column, af::dtype type) const;
    static std::pair<int8_t,int8_t> _dateDelimIndices(DateFormat format);
    static void _dateKeyToDate(af::array &out, DateFormat format);

public:
    explicit AFParser(char const *filename, char delimiter, bool hasHeader = false);
    explicit AFParser(std::string const &text, char delimiter);
    AFParser() = default;
    virtual ~AFParser();
    static std::string loadFile(char const *filename);


    void printData() const;

    /* Returns number of rows */
    unsigned long length() const { return _length; }
    /* Returns number of columns */
    unsigned long width() const { return _width; }

    af::array asDate(int column, DateFormat inputFormat) const;
    af::array asDateTime(int column, DateFormat format) const;

    af::array asUchar(int column) const;
    af::array asUshort(int column) const;
    af::array asShort(int column) const;

    af::array asUint(int column) const ;

    af::array asInt(int column) const ;

    af::array asFloat(int column) const ;

    af::array asU64(int column) const;

    af::array asS64(int column) const;

    af::array asDouble(int column) const;

    af::array asString(int column) const;

    af::array asTime(int column) const;

    af::array stringToBoolean(int column) const;
};


#endif /* AFParser_hpp */
