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
#include "Enums.h"

class AFParser {
private:
    std::unordered_map<std::string, unsigned long> _columnNames;
    af::array _data;
    af::array _indexer;
    char _delimiter;
    unsigned long _length;
    unsigned long _width;
    uint32_t* _maxColumnWidths;
    uint32_t _maxRowWidth;
    std::string _getString() const;
    void _generateIndexer();
    af::array _generateReversedCharacterIndices(int column) const;
public:
    explicit AFParser(char const *filename, char delimiter);
    AFParser() = default;
    virtual ~AFParser();
    static af::array findChar(char c, af::array const &txt);
    static af::array dateGenerator(uint16_t d = 0, uint16_t m = 0, uint16_t y = 0);
    static std::string loadFile(char const *filename);
    af::array asDate(int column, DateFormat inputFormat, bool isDelimited) const;
    af::array asDate(std::string column, DateFormat inputFormat, bool isDelimited);
    af::array asUnsigned32(int column);
    af::array asUnsigned32(std::string column);
    af::array asSigned32(int column);
    af::array asSigned32(std::string column);
    af::array asFloat(int column);
    af::array asFloat(std::string column);

    void nameColumn(std::string name, unsigned long idx);
    void nameColumn(std::string name, std::string old);
    void printRow(std::ostream& str, unsigned long row) const;
    void printColumn(std::ostream& str, unsigned long col) const;

    void stringMatch(unsigned int col, char const *match);
    /* Returns specific field in csv */
    std::string get(dim_t row, dim_t col) const;
    /* Returns number of rows */
    unsigned long length() const { return _length; }
    /* Returns number of columns */
    unsigned long width() const { return _width; }
};


#endif /* AFCSVParser_hpp */
