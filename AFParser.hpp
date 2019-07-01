//
//  AFParser.hpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#ifndef AFCSVParser_hpp
#define AFCSVParser_hpp

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

    void _generateIndexer(char delimiter);
    af::array _generateReversedCharacterIndices(int column) const;
    void _makeUniform(int column, af::array &output, af::array &negatives, af::array &points) const;
public:
    explicit AFParser(char const *filename, char delimiter);
    explicit AFParser(std::string const &text, char delimiter);
    AFParser() = default;
    virtual ~AFParser();
    static std::string loadFile(char const *filename);


    void printData() const;

    /* Returns number of rows */
    unsigned long length() const { return _length; }
    /* Returns number of columns */
    unsigned long width() const { return _width; }

    af::array asDate(int column, DateFormat inputFormat, bool isDelimited) const;

    af::array asUint(int column) const ;

    af::array asInt(int column) const ;

    af::array asFloat(int column) const ;

    af::array asUlong(int column) const;

    af::array asLong(int column) const;

    af::array asDouble(int column) const;

    af::array asString(int column) const;

    af::array stringToBoolean(int column) const;
};


#endif /* AFCSVParser_hpp */
