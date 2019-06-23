//
//  AFParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFParser.hpp"
#include "Utils.hpp"
#include "BatchFunctions.h"
#include <exception>
#include <sstream>
#include <utility>
#include <limits>

template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;
typedef unsigned long ulong;
using namespace af;
using namespace BatchFunctions;

AFParser::AFParser(char const *filename, char delimiter) {
    _length = 0;
    _width = 0;
    _maxRowWidth = 0;
    _maxColumnWidths = nullptr;
    _delimiter = delimiter;

    {
        std::string txt = loadFile(filename);
        _data = array(txt.size() + 1, txt.c_str());
        _data = _data(where(_data != '\r'));
    }
    _data.eval();
    _generateIndexer();
    sync();
}

AFParser::~AFParser() {
    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
}

array AFParser::findChar(char c, array const &txt) {
    return where(txt == (int)c);
}

std::string AFParser::loadFile(char const *filename) {
    std::ifstream file(filename);
    std::string text;
    file.seekg(0, std::ios::end);
    text.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    return text;
}

void AFParser::_generateIndexer() {

    _indexer = findChar('\n', _data);
    _length = _indexer.elements();
    {
        auto col_end = findChar(_delimiter, _data);
        _width = col_end.elements() / _length;
        col_end = moddims(col_end, _width++, _length);
        col_end = reorder(col_end, 1, 0);
        _indexer = join(1, col_end, _indexer);
    }

    {
        auto row_start = constant(0, 1, u32);
        row_start = join(0, row_start, _indexer.col(end).rows(0, end - 1) + 1);
        _indexer = join(1, row_start, _indexer);
    }
    _indexer.eval();
    auto tmp = max(diff1(_indexer,1),0);
    tmp -= join(1, constant(0, 1, u32), constant(1, dim4(1, tmp.dims(1) - 1), u32));
    _maxColumnWidths = tmp.host<uint32_t>();
    auto i = max(_indexer.col(end) - _indexer.col(0)).host<uint32_t>();
    _maxRowWidth = *i;
    freeHost(i);
}

array AFParser::dateGenerator(uint16_t d, uint16_t m, uint16_t y) {
    int date = y * 10000 + m * 100 + d;
    return array(1, 1, date);
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asDate(int const column, DateFormat inputFormat, bool isDelimited) const {
    int8_t const i = column != 0;
    int8_t const len = isDelimited ? 10 : 8;
    std::pair<int8_t, int8_t> year;
    std::pair<int8_t, int8_t> month;
    std::pair<int8_t, int8_t> day;

    switch (inputFormat) {
        case YYYYDDMM:
            year = std::pair<int8_t, int8_t>(0, 3);
            day = isDelimited ? std::pair<int8_t, int8_t>(5, 6) : std::pair<int8_t, int8_t>(4, 5);
            month = isDelimited ? std::pair<int8_t, int8_t>(8, 9) : std::pair<int8_t, int8_t>(7, 8);
            break;
        case YYYYMMDD:
            year = std::pair<int8_t, int8_t>(0, 3);
            month = isDelimited ? std::pair<int8_t, int8_t>(5, 6) : std::pair<int8_t, int8_t>(4, 5);
            day = isDelimited ? std::pair<int8_t, int8_t>(8, 9) : std::pair<int8_t, int8_t>(7, 8);
            break;
        case DDMMYYYY:
            year = isDelimited ? std::pair<int8_t, int8_t>(6, 9) : std::pair<int8_t, int8_t>(5, 8);
            day = std::pair<int8_t, int8_t>(0, 1);
            month = isDelimited ? std::pair<int8_t, int8_t>(3, 4) : std::pair<int8_t, int8_t>(2, 3);
            break;
        case MMDDYYYY:
            year = isDelimited ? std::pair<int8_t, int8_t>(6, 9) : std::pair<int8_t, int8_t>(5, 8);
            day = isDelimited ? std::pair<int8_t, int8_t>(3, 4) : std::pair<int8_t, int8_t>(2, 3);
            month = std::pair<int8_t, int8_t>(0, 1);
            break;
        default:
            throw std::invalid_argument("No such format!");
    }
    array out;
    {
        auto tmp = _indexer.col(column) + i;
        out = batchFunc(range(dim4(1, len), 1, u32), tmp, batchAdd);
    }
    out = moddims(_data(out), out.dims()) - '0';
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(out.dims(0),8));
    // matmul requires converting to f64 due to precision problems
    out = batchFunc(out, flip(pow(10,range(dim4(1,8), 1, u32)),1), batchMul);
    out = sum(out, 1);
    out.eval();

    return out;
}

/* This creates a copy of the column */
array AFParser::asDate(std::string column, DateFormat inputFormat, bool isDelimited) {
    return asDate(_columnNames[column], inputFormat, isDelimited);
}

/* generate chracter indices of a column invalid indices replaced with UINT32_MAX */
array AFParser::_generateReversedCharacterIndices(int const column) const {
    // Get the last character index
    unsigned int const i = column != 0;
    auto const maximum = _maxColumnWidths[column];
    auto out = _indexer.col(column + 1) - 1;
    // Get the indices of the whole number
    out = batchFunc(out, range(dim4(1, maximum), 1, u32), batchSub);
    // Removes the indices that do not point to part of the number (by pading these indices with UINT32_MAX)
    out(where(batchFunc(out, out.col(0), batchGreater))) = UINT32_MAX;
    out(where(batchFunc(out, _indexer.col(column) + i, batchLess))) = UINT32_MAX;
    // Transpose then flatten the array so that it can be used to index _data
    out = flat(reorder(out,1,0));
    out.eval();
    return out;
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asUnsigned32(int const column) {
    auto const maximum = _maxColumnWidths[column];

    auto out = _generateReversedCharacterIndices(column);
    // Scoped to force memory to clear
    {
        auto cond1 = where(out != UINT32_MAX);
        array j = out(cond1);
        j = _data(j);
        j = (j - '0').as(u32);
        j.eval();
        out(cond1) = j;
    }
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;
    // Multiply by powers of 10 and sum values across the row
    // Matmul can be used but requires casting between int and floats
    out = batchFunc(out, pow(10,range(dim4(1,maximum), 1, u32)), batchMul);
    out = sum(out, 1);
    out.eval();

    return out;
}

array AFParser::asUnsigned32(std::string column) {
    return asUnsigned32(_columnNames[column]);
}

array AFParser::asSigned32(int const column) {
    auto const maximum = _maxColumnWidths[column];
    auto out = _generateReversedCharacterIndices(column);

    array negatives;
    // Scoped to force memory to clear
    {
        auto cond = where(out != UINT32_MAX);
        array tmp = out(cond);
        tmp = _data(tmp);
        tmp = (tmp - '0').as(u32);
        tmp.eval();
        out = out;
        out(cond) = tmp;
        cond = where(out == '-' - '0');
        // cache negative rows
        negatives = cond / maximum;
        // set '-' to padded value
        out(cond) = 0;
    }
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;

    out = batchFunc(out.as(s32), pow(10,range(dim4(1,maximum), 1, s32)), batchMul);
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
    out.eval();
    return out;
}

array AFParser::asSigned32(std::string const column) {
    return asSigned32(_columnNames[column]);
}

array AFParser::asFloat(int const column) {
    unsigned int const i = column != 0;
    auto const maximum = _maxColumnWidths[column];
    auto out = _generateReversedCharacterIndices(column);
    array negatives;
    array points;

    // Scoped to force memory to clear
    {
        auto cond = where(out != UINT32_MAX);
        array tmp = out(cond);
        tmp = _data(tmp);
        tmp = (tmp - '0').as(u32);
        tmp.eval();
        out(cond) = tmp;
        cond = where(out == '-' - '0');
        // cache negative rows
        negatives = cond / maximum;
        // set '-' to padded value
        out(cond) = 0;
        cond = where(out == '.' - '0');
        // cache decimal points
        points = cond/maximum;
        points = join(1, points, (cond % maximum)).as(s32);
        out(cond) = 0;
    }
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;
    out = out.as(f32);
    out.eval();
    {
        auto exponent = range(dim4(1, maximum), 1, s32);
        exponent = batchFunc(exponent, points.col(1), batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(f32));
        out *= exponent;
    }
//    out = batchFunc(out, pow(10,range(dim4(1,maximum), 1, s32)), batchMul);
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
    out.eval();

    return out;
}

array AFParser::asFloat(std::string const column){
    return asFloat(_columnNames[column]);
}

std::string AFParser::get(dim_t row, dim_t col) const {
    auto idx = _indexer.host<unsigned int>();
    if (row > _length) throw std::invalid_argument("row index exceeds length");
    if (col > _width) throw std::invalid_argument("col index exceeds width");
    auto start = idx[col * _length + row] + !!col;
    auto length = idx[(col + 1) * _length + row] - start;
    freeHost(idx);
    return _getString().substr(start, length);
}
void AFParser::printRow(std::ostream& str, unsigned long row) const {
    auto idx = _indexer.host<unsigned int>();
    if (row > _length) throw std::invalid_argument("row index exceeds length");
    ulong start;
    ulong length;
    auto dat = _getString();
    for (unsigned int i = 0; i < _width; i++) {
        start = idx[i * _length + row] + (i != 0);
        length = idx[(i + 1) * _length + row] - start;
        str << dat.substr(start, length);
        if (i < _width - 1) str << ' ';
    }
    str << std::endl;
    freeHost(idx);
}
void AFParser::printColumn(std::ostream& str, unsigned long col) const {
    auto idx = _indexer.host<unsigned int>();
    if (col > _width) throw std::invalid_argument("col index exceeds width");
    ulong start;
    ulong length;
    auto dat = _getString();
    for (int i = 0; i < _length; i++) {
        start = idx[col * _length + i] + (col != 0);
        length = idx[(col + 1) * _length + i] - start;
        str << dat.substr(start, length);
        if (i < _length - 1) str << '\n';
    }
    str << std::endl;
    freeHost(idx);
}
std::string AFParser::_getString() const {
    auto str = _data.host<char>();
    auto out = std::string(str);
    freeHost(str);
    return out;
}

/* This function WILL change the original text array */
void AFParser::stringMatch(unsigned int col, char const *match) {
    auto const len = (unsigned int)strlen(match);
    unsigned int const offset = col != 0;

    {
        auto starts = _indexer.col(col) + offset;
        auto i = _indexer.col(col + 1) - starts;
        auto out = i == len; // Tests whether length matches the desired string length
        out = where(out); // Gets row indices where length matches
        out.eval();
        starts = starts(out); // get the character array indices
        starts.eval();

        i = batchFunc(starts, range(dim4(1, len), 1, u32), batchAdd);
        i = allTrue(batchFunc(moddims(_data(i), i.dims()), array(1, len, match), batchEqual), 1);
        i = where(i); // get row indices where the string matches
        i = out(i);
        i.eval();
        _length = i.elements();
        _indexer = _indexer(i, span);
    }

    auto tmp = batchFunc(range(dim4(1, _maxRowWidth), 1, u32), _indexer.col(0), batchAdd);
    tmp(where(batchFunc(tmp, _indexer.col(end), batchGreater))) = UINT32_MAX;
    tmp = flat(reorder(tmp,1,0));
    tmp = tmp(where(tmp != UINT32_MAX));
    tmp.eval();
    _data = join(0,_data(tmp),constant(0,1,_data.type()));
    _data.eval();

    _generateIndexer();
    sync();
}

void AFParser::nameColumn(std::string const name, unsigned long const idx) {
    _columnNames[name] = idx;
}

void AFParser::nameColumn(std::string const name, std::string const oldName) {
    _columnNames[name] = _columnNames[oldName];
    _columnNames.erase(oldName);
}